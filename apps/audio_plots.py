"""Interactive audio visualization for the EVA Streamlit app.

Renders a Plotly figure directly into a Streamlit tab without writing files.

Layout (dynamic — spectrograms are optional):
  Row 1        : audio_mixed waveform
  Row 2 (opt)  : audio_mixed spectrogram
  Row 3        : ElevenLabs waveform (only when elevenlabs_audio_recording.mp3 exists)
  Row 4 (opt)  : ElevenLabs spectrogram
  Row 5        : Speaker Turn Timeline

Waveform rendering:
  • Speaker segments are drawn in colour: blue = user, orange-red = assistant.
    Toggling a legend item hides all traces for that speaker.
  • Pause regions (speaker-change gaps) are drawn as shaded bands linked to the
    "Pause" legend item so they can be toggled on/off.
  • Only speaker-transition gaps are treated as pauses (consistent with turn_taking.py).

Turn data source (primary → fallback):
  1. metrics.json context  — the same MetricContext fields that turn_taking.py uses:
       context.audio_timestamps_user_turns / audio_timestamps_assistant_turns
         → dict[turn_id → list[[abs_start, abs_end]]]  (may be multi-segment)
       context.transcribed_*/intended_*_turns → dict[turn_id → str]
       latency_s = asst.segments[0][0] − user.segments[-1][1]  (per turn_id, same formula)
  2. elevenlabs_events.jsonl  — fallback when metrics.json is absent or has no timestamps:
       one turn per completed audio_start/audio_end session
       latency_s computed by temporal proximity (next assistant after this user)

X-axis range:
  Covers the longest of: audio_mixed duration, ElevenLabs audio duration, last turn end.
  Ensures neither audio file is clipped when they differ in length.

Spectrograms use a 4 kHz intermediate sample rate (_SPEC_SR) via librosa.resample so that:
  • frequency content up to 2 kHz (Nyquist) is preserved — representative of speech
  • heatmap size stays bounded (~60–250 K cells for 5–90 s recordings)
  • time axis (librosa.frames_to_time, t=0 origin) aligns with the waveform time axis
    (np.linspace(0, duration, n_samples), also t=0 origin)
"""

import json
import struct
import warnings
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pydub import AudioSegment

# =============================================================================
# Colours — visible in both Streamlit light and dark mode
# =============================================================================

USER_COLOR = "#4A90D9"  # mid-blue   — clear on white & dark
ASST_COLOR = "#E8724A"  # orange-red — clear on white & dark
USER_FILL = "rgba(74,144,217,0.22)"
ASST_FILL = "rgba(232,114,74,0.22)"
PAUSE_FILL = "rgba(140,140,140,0.18)"


# =============================================================================
# Turn data loading
# Primary: metrics.json context (MetricContext fields, same as turn_taking.py)
# Fallback: elevenlabs_events.jsonl (when metrics.json absent or has no timestamps)
# =============================================================================


def _load_metrics_context(record_dir: Path) -> dict | None:
    """Load metrics.json; return None if absent."""
    metrics_file = record_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def _build_turns_from_metrics(metrics_data: dict) -> list[dict] | None:
    """Build a turns list from MetricContext fields stored in metrics.json.

    Uses the exact same fields that turn_taking.py operates on:
      context.audio_timestamps_user_turns / audio_timestamps_assistant_turns
        → dict[turn_id → list[[abs_start, abs_end]]] — may have multiple
          segments per turn (e.g. interrupted turns)
      context.transcribed_*/intended_*_turns
        → dict[turn_id → str]

    Latency is computed directly from the timestamps using the same formula
    as turn_taking.py: asst.segments[0][0] − user.segments[-1][1]
    (first assistant segment start − last user segment end, per turn_id).

    Returns None when no timestamp data is present (falls back to EL log).
    """
    ctx = metrics_data.get("context") or {}
    user_ts = ctx.get("audio_timestamps_user_turns") or {}
    asst_ts = ctx.get("audio_timestamps_assistant_turns") or {}
    if not user_ts and not asst_ts:
        return None

    transcribed_user = ctx.get("transcribed_user_turns") or {}
    transcribed_asst = ctx.get("transcribed_assistant_turns") or {}
    intended_user = ctx.get("intended_user_turns") or {}
    intended_asst = ctx.get("intended_assistant_turns") or {}

    # Reference time: earliest timestamp across all turns (same as turn_taking.py)
    all_starts = [segs[0][0] for segs in list(user_ts.values()) + list(asst_ts.values()) if segs]
    t0 = min(all_starts) if all_starts else 0.0

    def _rel(segs: list) -> list[tuple[float, float]]:
        return [(s - t0, e - t0) for s, e in segs] if segs else []

    turns: list[dict] = []

    for tid_str, segs in asst_ts.items():
        if not segs:
            continue
        rel = _rel(segs)
        turns.append(
            {
                "turn_id": int(tid_str),
                "speaker": "assistant",
                "segments": rel,
                "start": rel[0][0],
                "end": rel[-1][1],
                "duration": rel[-1][1] - rel[0][0],
                "transcript_heard": transcribed_asst.get(tid_str, ""),
                "transcript_intended": intended_asst.get(tid_str, ""),
                "latency_s": None,
            }
        )

    for tid_str, segs in user_ts.items():
        if not segs:
            continue
        rel = _rel(segs)
        # Latency: same formula as turn_taking.py — asst first-seg start − user last-seg end.
        # Uses the matching assistant turn (same turn_id); None if no assistant turn exists.
        a_segs = asst_ts.get(tid_str)
        latency_s = round(a_segs[0][0] - segs[-1][1], 6) if a_segs else None
        turns.append(
            {
                "turn_id": int(tid_str),
                "speaker": "user",
                "segments": rel,
                "start": rel[0][0],
                "end": rel[-1][1],
                "duration": rel[-1][1] - rel[0][0],
                "transcript_heard": transcribed_user.get(tid_str, ""),
                "transcript_intended": intended_user.get(tid_str, ""),
                "latency_s": latency_s,
            }
        )

    turns.sort(key=lambda t: t["start"])
    return turns


def _parse_elevenlabs_events(events_file: Path) -> list[dict]:
    """Parse elevenlabs_events.jsonl into a flat list of audio-session turns.

    Each completed audio_start/audio_end pair for a participant becomes one
    turn dict.  Turn IDs are sequential integers across all participants (not
    per-speaker).  Transcripts and latencies are left empty here and filled in
    by _patch_fallback_transcripts and _compute_and_patch_latencies.

    Speaker assignment:
      event["user"] == "pipecat_agent"  → speaker = "assistant"
      anything else                     → speaker = "user" (EL user-simulator)

    Time reference:
      t0 = earliest audio_timestamp across all completed sessions.
      All start/end values stored as relative seconds from t0.
    """
    events = []
    with open(events_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    audio_events = [e for e in events if e.get("event_type") in ("audio_start", "audio_end")]
    audio_events.sort(key=lambda x: x.get("audio_timestamp", 0))

    active: dict = {}
    raw: list = []
    for ev in audio_events:
        user = ev.get("user")
        etype = ev.get("event_type")
        ts = ev.get("audio_timestamp")
        if etype == "audio_start":
            if user not in active or active[user].get("end") is not None:
                active[user] = {"user": user, "start": ts, "end": None}
        elif etype == "audio_end":
            if user in active and active[user].get("end") is None:
                active[user]["end"] = ts
                active[user]["duration"] = ts - active[user]["start"]
                raw.append(active[user].copy())

    raw.sort(key=lambda x: x["start"])
    t0 = min((t["start"] for t in raw), default=0.0)

    user_idx = asst_idx = 0
    turns: list[dict] = []
    for i, t in enumerate(raw):
        is_asst = t["user"] == "pipecat_agent"
        speaker = "assistant" if is_asst else "user"
        s_rel = t["start"] - t0
        e_rel = t["end"] - t0
        turns.append(
            {
                "turn_id": i,
                "speaker": speaker,
                "segments": [(s_rel, e_rel)],
                "start": s_rel,
                "end": e_rel,
                "duration": t.get("duration", e_rel - s_rel),
                "transcript_heard": "",
                "transcript_intended": "",
                "latency_s": None,
                "_seq_idx": asst_idx if is_asst else user_idx,
            }
        )
        if is_asst:
            asst_idx += 1
        else:
            user_idx += 1
    return turns


def _patch_fallback_transcripts(turns: list[dict], transcript_file: Path) -> None:
    """Fill transcript fields in EL-log turns from transcript.jsonl.

    Matches transcripts to turns by sequential order per speaker role
    (first user turn gets user transcript[0], second gets [1], etc.).
    Called after _parse_elevenlabs_events, before _compute_and_patch_latencies.
    """
    tx: dict[str, list[str]] = {"user": [], "assistant": []}
    if transcript_file.exists():
        with open(transcript_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    role = entry.get("type", "")
                    content = entry.get("content", "")
                    if role in tx:
                        tx[role].append(content)
    for turn in turns:
        idx = turn.pop("_seq_idx", 0)
        key = "assistant" if turn["speaker"] == "assistant" else "user"
        text = tx[key][idx] if idx < len(tx[key]) else ""
        turn["transcript_heard"] = text
        turn["transcript_intended"] = text


def _compute_and_patch_latencies(turns: list[dict]) -> None:
    """Compute per-user-turn response latency and patch in-place.

    Formula (identical to turn_taking.py _compute_per_turn_latency_and_timing_labels):
      latency_s = asst.segments[0][0] - user.segments[-1][1]
                  (first assistant segment start − last user segment end)

    Matching: turn_taking.py matches by shared turn_id; here we match by
    temporal proximity (next assistant turn after this user turn in time order)
    — equivalent for linear conversations.
    """
    sorted_turns = sorted(turns, key=lambda t: t["start"])
    for i, turn in enumerate(sorted_turns):
        if turn["speaker"] != "user":
            continue
        for j in range(i + 1, len(sorted_turns)):
            if sorted_turns[j]["speaker"] == "assistant":
                latency_s = sorted_turns[j]["segments"][0][0] - turn["segments"][-1][1]
                turn["latency_s"] = round(latency_s, 6)
                break


def _calculate_pauses(turns_rel: list[dict]) -> list[dict]:
    """Compute speaker-transition gaps, consistent with turn_taking.py.

    Only gaps where the speaker changes (user→assistant or assistant→user)
    are counted — mirroring the `if prev_role != next_role` guard in
    turn_taking.py _format_conversation_context (lines 81-86).

    Same-speaker consecutive segments (e.g. two user audio sessions back to
    back) are ignored, as turn_taking.py does not treat these as pauses.
    """
    all_segs = sorted(
        [(s, e, turn["speaker"]) for turn in turns_rel for s, e in turn["segments"]],
        key=lambda x: x[0],
    )
    pauses = []
    for i in range(len(all_segs) - 1):
        from_spk = all_segs[i][2]
        to_spk = all_segs[i + 1][2]
        if from_spk == to_spk:
            continue  # same-speaker gap — not a turn-taking transition
        cur_end = all_segs[i][1]
        nxt_start = all_segs[i + 1][0]
        gap = nxt_start - cur_end
        if gap > 0.001:
            pauses.append(
                {
                    "from_speaker": from_spk,
                    "to_speaker": to_spk,
                    "start": cur_end,
                    "end": nxt_start,
                    "duration_seconds": gap,
                }
            )
    return pauses


# =============================================================================
# Audio loading helpers
# =============================================================================


def _wav_actual_n_samples(path: Path) -> tuple[int, int, int] | None:
    """Return (n_samples_per_channel, sr, sample_width) from actual WAV file bytes.

    Scans the RIFF chunks to find the fmt and data chunks.  The data chunk size
    is derived from (file_size − data_chunk_start) rather than from the header
    field, which is frequently wrong when the recorder fails to update it.

    Returns None for non-WAV files or unreadable files.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                return None
            f.read(4)  # RIFF chunk size — unreliable, ignore
            if f.read(4) != b"WAVE":
                return None
            sr = channels = sample_width = None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid = hdr[:4]
                csz = struct.unpack("<I", hdr[4:])[0]
                if csz > 100_000_000:
                    break  # sanity guard against corrupt chunk sizes
                if cid == b"fmt ":
                    raw = f.read(min(csz, 16))
                    _, channels, sr, _, _, bits = struct.unpack_from("<HHIIHH", raw)
                    sample_width = bits // 8
                    # skip remaining fmt bytes + RIFF pad (chunks are even-aligned)
                    skip = (csz - min(csz, 16)) + (csz % 2)
                    if skip > 0:
                        f.seek(skip, 1)
                elif cid == b"data":
                    if sr and channels and sample_width:
                        data_start = f.tell()
                        file_size = path.stat().st_size
                        actual_bytes = file_size - data_start
                        n_samples = actual_bytes // (sample_width * channels)
                        return n_samples, sr, sample_width
                    break
                else:
                    # RIFF chunks are padded to even byte boundaries
                    f.seek(csz + (csz % 2), 1)
    except Exception:
        pass
    return None


def _load_pydub(path: Path) -> tuple:
    seg = AudioSegment.from_file(str(path))
    n_channels = seg.channels
    if n_channels > 1:
        seg = seg.set_channels(1)
    sr = seg.frame_rate
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0

    # WAV files from some recorders have an incorrect data-chunk size in the
    # header (written before the call starts, never updated when it ends).
    # Cross-check against the real file size and reload raw PCM if the
    # declared duration is more than 5% shorter than the actual file content.
    if path.suffix.lower() == ".wav":
        info = _wav_actual_n_samples(path)
        if info is not None:
            n_actual, _, sw = info
            dur_declared = len(y) / sr
            dur_actual = n_actual / sr
            if dur_actual > dur_declared + 1.0:
                try:
                    dtype = np.int16 if sw == 2 else np.int32
                    divisor = 32768.0 if sw == 2 else 2_147_483_648.0
                    with open(path, "rb") as f:
                        # Re-seek to the data chunk start (RIFF + size + WAVE)
                        f.read(4)
                        f.read(4)
                        f.read(4)
                        while True:
                            hdr = f.read(8)
                            if len(hdr) < 8:
                                break
                            cid = hdr[:4]
                            csz = struct.unpack("<I", hdr[4:])[0]
                            if cid == b"data":
                                raw = np.frombuffer(f.read(), dtype=dtype).astype(np.float32) / divisor
                                if n_channels > 1:
                                    raw = raw[: (len(raw) // n_channels) * n_channels]
                                    raw = raw.reshape(-1, n_channels).mean(axis=1)
                                y = raw
                                break
                            if csz > 100_000_000:
                                break
                            f.seek(csz + (csz % 2), 1)
                except Exception:
                    pass  # keep pydub result

    return y, sr


def _load_librosa(path: Path) -> tuple:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*audioread.*")
        return librosa.load(str(path), sr=None, mono=True)


def _downsample(y: np.ndarray, sr: float, target_rate: int = 100) -> tuple:
    duration = len(y) / sr
    target = max(2, int(duration * target_rate))
    if len(y) > target:
        step = max(1, len(y) // target)
        y_ds = y[::step]
        sr_ds = sr * len(y_ds) / len(y)
    else:
        y_ds, sr_ds = y, sr
    return y_ds, sr_ds


def _wrap(text: str, width: int = 80) -> str:
    words = text.split()
    lines, current, length = [], [], 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


# =============================================================================
# Spectrogram parameters
# =============================================================================

# Intermediate sample rate used for spectrogram computation.
# 4 kHz preserves speech content up to 2 kHz (Nyquist) while keeping the
# heatmap to roughly 60–250K cells for typical 5–90 s recordings.
_SPEC_SR = 4000  # Hz
_SPEC_N_FFT = 512  # → 257 freq bins, 7.8 Hz resolution
_SPEC_HOP = 512  # → ~0.128 s/frame at 4 kHz


# =============================================================================
# Data preparation
# =============================================================================


def _prepare_data(record_dir: Path) -> dict:
    audio_mixed = next(record_dir.glob("audio_mixed*.wav"), record_dir / "audio_mixed.wav")
    audio_el = record_dir / "elevenlabs_audio_recording.mp3"
    events_file = record_dir / "elevenlabs_events.jsonl"
    transcript = record_dir / "transcript.jsonl"

    # --- Turn data ---
    # Primary: metrics.json context — same fields turn_taking.py uses, with
    # multi-segment turns, matched transcripts, and turn_id-based latency.
    # Fallback: elevenlabs_events.jsonl — one entry per audio session, latency
    # computed by temporal proximity when metrics.json is absent.
    turns_rel: list[dict] = []
    metrics_data = _load_metrics_context(record_dir)
    if metrics_data:
        built = _build_turns_from_metrics(metrics_data)
        if built:
            turns_rel = built

    if not turns_rel and events_file.exists():
        turns_rel = _parse_elevenlabs_events(events_file)
        _patch_fallback_transcripts(turns_rel, transcript)
        _compute_and_patch_latencies(turns_rel)

    pauses_rel = _calculate_pauses(turns_rel)

    # --- Audio: mixed ---
    y_mixed, sr_mixed, duration, mixed_loaded = None, None, 0.0, False
    if audio_mixed.exists():
        try:
            y_mixed, sr_mixed = _load_pydub(audio_mixed)
            duration = len(y_mixed) / sr_mixed
            mixed_loaded = True
        except Exception:
            pass

    if mixed_loaded:
        y_ds, _ = _downsample(y_mixed, sr_mixed)
        t_mixed = np.linspace(0, duration, len(y_ds))
    else:
        y_ds = np.array([])
        t_mixed = np.array([])

    # --- Audio: ElevenLabs ---
    el_y_ds, el_t, el_spec = np.array([]), np.array([]), None
    el_loaded = False
    el_duration = 0.0
    if audio_el.exists():
        try:
            _el_y, _el_sr = _load_librosa(audio_el)
            el_y_ds, _ = _downsample(_el_y, _el_sr)
            el_duration = len(_el_y) / _el_sr
            el_t = np.linspace(0, el_duration, len(el_y_ds))
            el_loaded = True
            # Spectrogram: resample to _SPEC_SR (4 kHz) for speech-range content.
            # x axis pinned to el_duration via np.linspace so it aligns with el_t.
            try:
                _el_y_spec = librosa.resample(_el_y, orig_sr=_el_sr, target_sr=_SPEC_SR)
                D = librosa.amplitude_to_db(
                    np.abs(librosa.stft(_el_y_spec, hop_length=_SPEC_HOP, n_fft=_SPEC_N_FFT)), ref=np.max
                )
                freqs = librosa.fft_frequencies(sr=_SPEC_SR, n_fft=_SPEC_N_FFT)
                times = np.linspace(0, el_duration, D.shape[1])
                el_spec = (D, freqs, times)
            except Exception:
                pass
        except Exception:
            pass

    # x-axis: audio file durations only.
    # turns_end is excluded — turn timestamps can exceed the recording length
    # and would push the axis beyond the actual audio.
    plot_xlim = [0, max(duration if mixed_loaded else 0.0, el_duration, 1.0)]

    # --- Spectrogram: mixed ---
    # x axis pinned to `duration` via np.linspace so it aligns exactly with
    # t_mixed. STFT data is unchanged; only the frame→time mapping differs from
    # frames_to_time by at most one hop (128 ms), which is visually imperceptible.
    mixed_spec = None
    if mixed_loaded:
        try:
            _y_spec = librosa.resample(y_mixed, orig_sr=sr_mixed, target_sr=_SPEC_SR)
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(_y_spec, hop_length=_SPEC_HOP, n_fft=_SPEC_N_FFT)), ref=np.max
            )
            freqs = librosa.fft_frequencies(sr=_SPEC_SR, n_fft=_SPEC_N_FFT)
            times = np.linspace(0, duration, D.shape[1])
            mixed_spec = (D, freqs, times)
        except Exception:
            pass

    return {
        "duration": duration,
        "plot_xlim": plot_xlim,
        "mixed_loaded": mixed_loaded,
        "y_ds": y_ds,
        "t_mixed": t_mixed,
        "el_loaded": el_loaded,
        "el_y_ds": el_y_ds,
        "el_t": el_t,
        "mixed_spec": mixed_spec,
        "el_spec": el_spec,
        "turns_rel": turns_rel,
        "pauses_rel": pauses_rel,
    }


# =============================================================================
# Plotly figure builder
# =============================================================================


def _build_figure(
    data: dict, show_mixed_spec: bool = False, show_el_spec: bool = False, title_suffix: str = ""
) -> go.Figure:

    turns_rel = data["turns_rel"]
    pauses_rel = data["pauses_rel"]
    plot_xlim = data["plot_xlim"]

    # ------------------------------------------------------------------ #
    # Dynamic row layout
    # ------------------------------------------------------------------ #
    row_keys: list[str] = ["mixed_waveform"]
    if show_mixed_spec and data["mixed_spec"]:
        row_keys.append("mixed_spec")
    if data["el_loaded"]:
        row_keys.append("el_waveform")
        if show_el_spec and data["el_spec"]:
            row_keys.append("el_spec")
    row_keys.append("timeline")

    _titles = {
        "mixed_waveform": "Waveform \u2014 audio_mixed.wav",
        "mixed_spec": "Spectrogram \u2014 audio_mixed.wav",
        "el_waveform": "Waveform \u2014 elevenlabs_audio_recording.mp3",
        "el_spec": "Spectrogram \u2014 elevenlabs_audio_recording.mp3",
        "timeline": "Speaker Turn Timeline",
    }
    _heights = {
        "mixed_waveform": 1.5,
        "mixed_spec": 1.3,
        "el_waveform": 1.5,
        "el_spec": 1.3,
        "timeline": 1.5,
    }

    n_rows = len(row_keys)
    row_of = {k: i + 1 for i, k in enumerate(row_keys)}
    row_heights = [_heights[k] for k in row_keys]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[_titles[k] for k in row_keys],
        row_heights=row_heights,
        vertical_spacing=0.05,
    )

    fig.update_layout(
        title={
            "text": f"Speaker Turn Analysis \u2014 Pause Detection{title_suffix}",
            "font": {"size": 15},
        },
        height=max(700, 420 * n_rows),
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bordercolor": "rgba(128,128,128,0.4)",
            "borderwidth": 1,
        },
    )

    # ------------------------------------------------------------------ #
    # Centralised legend — one dummy trace per category, added once.
    # All real traces use showlegend=False + legendgroup for toggling.
    # ------------------------------------------------------------------ #
    for _name, _color, _symbol in [
        ("User", USER_COLOR, "square"),
        ("Assistant", ASST_COLOR, "square"),
        ("Pause", "rgba(140,140,140,0.40)", "square"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": _color, "size": 12, "symbol": _symbol, "line": {"color": _color, "width": 2}},
                name=_name,
                legendgroup=_name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # ------------------------------------------------------------------ #
    # Hover text — per-sample transcript strings, keyed by turn segment
    # ------------------------------------------------------------------ #
    def _hover_texts(time_array: np.ndarray) -> list:
        if len(time_array) == 0:
            return []
        texts = np.full(len(time_array), "", dtype=object)

        for turn in turns_rel:
            speaker = "Assistant" if turn["speaker"] == "assistant" else "User"
            transcript = turn["transcript_heard"] or turn["transcript_intended"] or "(no transcript)"

            latency_line = ""
            if turn["speaker"] == "user" and turn.get("latency_s") is not None:
                latency_line = f"<br>Response latency:\u00a0{turn['latency_s'] * 1000:.0f}\u00a0ms"

            hover = (
                f"<b>Turn\u00a0{turn['turn_id']}\u00a0\u2014\u00a0{speaker}</b><br>"
                f"t\u00a0=\u00a0{turn['start']:.2f}s\u2013{turn['end']:.2f}s "
                f"({turn['duration']:.1f}s)" + latency_line + f"<br><br>{_wrap(transcript)}"
            )

            for seg_s, seg_e in turn["segments"]:
                mask = (time_array >= seg_s) & (time_array <= seg_e)
                texts[mask] = hover

        for pause in pauses_rel:
            hover = (
                f"<b>Pause</b><br>"
                f"t\u00a0=\u00a0{pause['start']:.2f}s\u2013{pause['end']:.2f}s<br>"
                f"Duration:\u00a0{pause['duration_seconds'] * 1000:.0f}\u00a0ms<br>"
                f"{pause['from_speaker']}\u00a0\u2192\u00a0{pause['to_speaker']}"
            )
            mask = (time_array >= pause["start"]) & (time_array <= pause["end"])
            texts[mask] = hover

        return texts.tolist()

    # ------------------------------------------------------------------ #
    # Colour-coded waveform
    # Speaker segments — blue (user) / orange-red (assistant).
    # Pause shaded bands — linked to "Pause" legend toggle.
    # X-axis range is set by plot_xlim (independent of trace data extent).
    # ------------------------------------------------------------------ #
    def _colored_waveform(
        row: int, y: np.ndarray, t: np.ndarray, y_range: list, speaker_filter: set[str] | None = None
    ) -> None:
        if len(y) == 0:
            fig.add_annotation(
                text="No file available",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"color": "gray", "size": 11},
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="Amplitude", range=[-1.0, 1.0], row=row, col=1)
            return

        # Flat list of speaker audio segments, sorted by start time.
        visible_turns = [turn for turn in turns_rel if speaker_filter is None or turn["speaker"] in speaker_filter]
        all_segs = sorted(
            [
                (s, e, "asst" if turn["speaker"] == "assistant" else "user")
                for turn in visible_turns
                for s, e in turn["segments"]
            ],
            key=lambda s: s[0],
        )

        _color_map = {"user": USER_COLOR, "asst": ASST_COLOR}
        _name_map = {"user": "User", "asst": "Assistant"}

        for seg_s, seg_e, spk in all_segs:
            mask = (t >= seg_s) & (t <= seg_e)
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=t[mask].tolist(),
                    y=y[mask].tolist(),
                    mode="lines",
                    line={"width": 1.0, "color": _color_map[spk]},
                    opacity=0.85,
                    name=_name_map[spk],
                    legendgroup=_name_map[spk],
                    showlegend=False,
                    text=_hover_texts(t[mask]),
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=row,
                col=1,
            )

        # Pause shaded bands — Scatter traces so they toggle with the legend.
        y0, y1 = y_range[0], y_range[1]
        for pause in pauses_rel:
            fig.add_trace(
                go.Scatter(
                    x=[pause["start"], pause["end"], pause["end"], pause["start"], pause["start"]],
                    y=[y1, y1, y0, y0, y1],
                    fill="toself",
                    fillcolor=PAUSE_FILL,
                    line={"width": 0},
                    mode="lines",
                    name="Pause",
                    legendgroup="Pause",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text="Amplitude", range=y_range, row=row, col=1)

    # ------------------------------------------------------------------ #
    # Spectrogram row
    # Layer 1: Heatmap (STFT at _SPEC_SR=4 kHz, 0–2 kHz Nyquist).
    # Layer 2: Invisible transcript strip for hover tooltips.
    # Layer 3: Speaker turn vrects (user / assistant fill colours).
    # Layer 4: Pause shaded bands — linked to "Pause" legend toggle.
    # ------------------------------------------------------------------ #
    def _spec_row(row: int, spec: tuple, label: str, speaker_filter: set[str] | None = None) -> None:
        D, freqs, times = spec

        fig.add_trace(
            go.Heatmap(
                z=D,
                x=times,
                y=freqs,
                colorscale="Viridis",
                zmin=-80,
                zmax=0,
                colorbar={"title": "dB", "thickness": 12, "len": 0.12, "x": 1.01},
                hovertemplate=("t=%{x:.2f}s  freq=%{y:.0f}Hz  %{z:.1f}dB<extra>" + label + "</extra>"),
                showscale=True,
            ),
            row=row,
            col=1,
        )

        # Invisible transcript strip at freq_max
        strip_t = np.asarray(times, dtype=float)
        freq_max = float(freqs[-1])
        fig.add_trace(
            go.Scatter(
                x=strip_t.tolist(),
                y=[freq_max] * len(strip_t),
                mode="markers",
                marker={"opacity": 0, "size": 6},
                showlegend=False,
                name="",
                text=_hover_texts(strip_t),
                hovertemplate="%{text}<extra>Transcript</extra>",
            ),
            row=row,
            col=1,
        )

        # Pause shaded bands — Scatter traces so they toggle with the legend.
        f0, f1 = float(freqs[0]), float(freqs[-1])
        for pause in pauses_rel:
            fig.add_trace(
                go.Scatter(
                    x=[pause["start"], pause["end"], pause["end"], pause["start"], pause["start"]],
                    y=[f1, f1, f0, f0, f1],
                    fill="toself",
                    fillcolor=PAUSE_FILL,
                    line={"width": 0},
                    mode="lines",
                    name="Pause",
                    legendgroup="Pause",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text="Freq (Hz)", row=row, col=1)

    def _no_file(row: int) -> None:
        fig.add_annotation(
            text="No file available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": "gray", "size": 11},
            row=row,
            col=1,
        )

    # ---- Mixed waveform ----
    if data["mixed_loaded"] and len(data["y_ds"]) > 0:
        y_range = [float(data["y_ds"].min() * 1.1), float(data["y_ds"].max() * 1.1)]
        _colored_waveform(row_of["mixed_waveform"], data["y_ds"], data["t_mixed"], y_range)
    else:
        _no_file(row_of["mixed_waveform"])
        fig.update_yaxes(title_text="Amplitude", range=[-1.0, 1.0], row=row_of["mixed_waveform"], col=1)

    # ---- Mixed spectrogram (optional) ----
    if "mixed_spec" in row_of:
        if data["mixed_spec"]:
            _spec_row(row_of["mixed_spec"], data["mixed_spec"], "Mixed Spec")
        else:
            _no_file(row_of["mixed_spec"])
            fig.update_yaxes(title_text="Freq (Hz)", row=row_of["mixed_spec"], col=1)

    # ---- ElevenLabs waveform (only present when el_loaded=True) ----
    # No speaker_filter: turn times from the EL log cover both user and assistant,
    # so both get colour-coded identically to the mixed waveform.  Assistant
    # regions will show a flat/silent waveform since the EL file only captures
    # the user-simulator's outgoing audio, which is expected.
    if "el_waveform" in row_of:
        if len(data["el_y_ds"]) > 0:
            el_range = [float(data["el_y_ds"].min() * 1.1), float(data["el_y_ds"].max() * 1.1)]
            _colored_waveform(row_of["el_waveform"], data["el_y_ds"], data["el_t"], el_range)
        else:
            _no_file(row_of["el_waveform"])
            fig.update_yaxes(title_text="Amplitude", range=[-1.0, 1.0], row=row_of["el_waveform"], col=1)

    # ---- ElevenLabs spectrogram (optional) ----
    if "el_spec" in row_of:
        if data["el_spec"]:
            _spec_row(row_of["el_spec"], data["el_spec"], "EL Spec")
        else:
            _no_file(row_of["el_spec"])
            fig.update_yaxes(title_text="Freq (Hz)", row=row_of["el_spec"], col=1)

    # ------------------------------------------------------------------ #
    # Speaker Turn Timeline
    # ------------------------------------------------------------------ #
    tl_row = row_of["timeline"]

    for turn in turns_rel:
        is_asst = turn["speaker"] == "assistant"
        speaker = "Assistant" if is_asst else "User"
        y_pos = 2.0 if is_asst else 1.0
        bar_fill = "rgba(232,114,74,0.80)" if is_asst else "rgba(74,144,217,0.80)"
        bar_line = "rgba(180,70,30,1)" if is_asst else "rgba(30,90,170,1)"

        transcript = turn["transcript_heard"] or turn["transcript_intended"] or "(no transcript)"
        latency_line = ""
        if not is_asst and turn.get("latency_s") is not None:
            latency_line = f"<br>Response latency:\u00a0{turn['latency_s'] * 1000:.0f}\u00a0ms"

        hover = (
            f"<b>Turn\u00a0{turn['turn_id']}\u00a0\u2014\u00a0{speaker}</b><br>"
            f"t\u00a0=\u00a0{turn['start']:.2f}s\u2013{turn['end']:.2f}s "
            f"({turn['duration']:.1f}s)" + latency_line + f"<br><br>{_wrap(transcript)}"
        )

        # Visual bars — one per segment (handles multi-segment interrupted turns)
        for seg_s, seg_e in turn["segments"]:
            fig.add_trace(
                go.Scatter(
                    x=[seg_s, seg_e, seg_e, seg_s, seg_s],
                    y=[y_pos - 0.38, y_pos - 0.38, y_pos + 0.38, y_pos + 0.38, y_pos - 0.38],
                    fill="toself",
                    fillcolor=bar_fill,
                    line={"color": bar_line, "width": 1},
                    mode="lines",
                    hoverinfo="skip",
                    name=speaker,
                    legendgroup=speaker,
                    showlegend=False,
                ),
                row=tl_row,
                col=1,
            )

        # Dense hover strip across full turn envelope (~2 pts/sec, min 5)
        n_pts = max(5, int(turn["duration"] * 2))
        x_strip = np.linspace(turn["start"], turn["end"], n_pts).tolist()
        fig.add_trace(
            go.Scatter(
                x=x_strip,
                y=[y_pos] * n_pts,
                mode="markers",
                marker={"opacity": 0, "size": 10},
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
                name="",
            ),
            row=tl_row,
            col=1,
        )

        # Duration label on the first (or only) segment
        seg0_s, seg0_e = turn["segments"][0]
        fig.add_annotation(
            x=seg0_s + (seg0_e - seg0_s) / 2,
            y=y_pos,
            text=f"T{turn['turn_id']}\u00a0{turn['duration']:.1f}s",
            showarrow=False,
            font={"size": 8, "color": "white"},
            xref=f"x{tl_row}",
            yref=f"y{tl_row}",
        )

    # Latency arrows: user last-segment-end → assistant first-segment-start
    user_by_id = {t["turn_id"]: t for t in turns_rel if t["speaker"] == "user"}
    asst_by_id = {t["turn_id"]: t for t in turns_rel if t["speaker"] == "assistant"}
    for tid, user_turn in user_by_id.items():
        if not user_turn.get("latency_s") or user_turn["latency_s"] <= 0.05:
            continue
        asst_turn = asst_by_id.get(tid)
        if asst_turn is None:
            continue
        user_end = user_turn["segments"][-1][1]
        asst_start = asst_turn["segments"][0][0]
        if asst_start <= user_end:
            continue
        fig.add_annotation(
            x=(user_end + asst_start) / 2,
            y=1.5,
            text=f"\u2194\u00a0{user_turn['latency_s'] * 1000:.0f}ms",
            showarrow=False,
            font={"size": 7, "color": "dimgray"},
            bgcolor="rgba(255,255,255,0.7)",
            xref=f"x{tl_row}",
            yref=f"y{tl_row}",
        )

    # Pause boxes on timeline
    for pause in pauses_rel:
        hover = (
            f"<b>Pause</b><br>"
            f"t\u00a0=\u00a0{pause['start']:.2f}s\u2013{pause['end']:.2f}s<br>"
            f"Duration:\u00a0{pause['duration_seconds'] * 1000:.0f}\u00a0ms<br>"
            f"{pause['from_speaker']}\u00a0\u2192\u00a0{pause['to_speaker']}"
        )
        fig.add_trace(
            go.Scatter(
                x=[pause["start"], pause["end"], pause["end"], pause["start"], pause["start"]],
                y=[1.15, 1.15, 1.85, 1.85, 1.15],
                fill="toself",
                fillcolor="rgba(140,140,140,0.40)",
                line={"color": "rgba(180,60,60,0.8)", "width": 1, "dash": "dash"},
                mode="lines",
                hoverinfo="skip",
                name="Pause",
                legendgroup="Pause",
                showlegend=False,
            ),
            row=tl_row,
            col=1,
        )

        n_pts = max(5, int(pause["duration_seconds"] * 2))
        x_strip = np.linspace(pause["start"], pause["end"], n_pts).tolist()
        fig.add_trace(
            go.Scatter(
                x=x_strip,
                y=[1.5] * n_pts,
                mode="markers",
                marker={"opacity": 0, "size": 10},
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
                name="",
            ),
            row=tl_row,
            col=1,
        )

        fig.add_annotation(
            x=pause["start"] + pause["duration_seconds"] / 2,
            y=1.5,
            text=f"{pause['duration_seconds'] * 1000:.0f}ms",
            showarrow=False,
            font={"size": 7, "color": "dimgray"},
            bgcolor="rgba(255,255,255,0.7)",
            xref=f"x{tl_row}",
            yref=f"y{tl_row}",
        )

    fig.update_yaxes(
        tickvals=[1, 2],
        ticktext=["User", "Assistant"],
        range=[0.5, 2.5],
        title_text="Speaker",
        row=tl_row,
        col=1,
    )
    fig.update_xaxes(title_text="Time (seconds)", row=tl_row, col=1)

    # Shared x-range + grid for all rows.
    # showticklabels=True is required on every row because shared_xaxes=True
    # hides tick labels on all but the bottom subplot by default.
    for r in range(1, n_rows + 1):
        fig.update_xaxes(
            range=plot_xlim,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            showticklabels=True,
            nticks=20,
            row=r,
            col=1,
        )
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", row=r, col=1)

    return fig


# =============================================================================
# Streamlit caching — module-level so the cache persists across reruns
# =============================================================================


def _audio_mtime(record_dir: Path) -> int:
    """Return the most-recent mtime (seconds) of audio files in record_dir.

    Included in the cache key so the cache is invalidated when a file changes
    — e.g. when a new recording replaces a shorter one, or when the WAV was
    still being written when preload_audio_data() was first called.
    """
    audio_mixed = next(record_dir.glob("audio_mixed*.wav"), record_dir / "audio_mixed.wav")
    audio_el = record_dir / "elevenlabs_audio_recording.mp3"
    mtime = 0
    for p in (audio_mixed, audio_el):
        if p.exists():
            mtime = max(mtime, int(p.stat().st_mtime))
    return mtime


@st.cache_data(show_spinner="Loading audio files\u2026")
def _cache_audio_data(path_str: str, audio_mtime: int = 0) -> dict:
    """Cache the heavy data-loading step (file I/O + spectrogram computation).

    Keyed on the record directory path AND the audio file mtime so the cache
    is automatically invalidated when the audio files change.
    _build_figure() is fast and runs on each rerun with the pre-loaded data.
    """
    return _prepare_data(Path(path_str))


def preload_audio_data(record_dir: Path) -> None:
    """Warm the audio-data cache for *record_dir*.

    Call this before the tab widgets are rendered so the heavy I/O happens
    while the rest of the page is being built, rather than on first tab open.
    Silently skips records that have no audio files.
    """
    events_file = record_dir / "elevenlabs_events.jsonl"
    audio_mixed = next(record_dir.glob("audio_mixed*.wav"), record_dir / "audio_mixed.wav")
    if events_file.exists() or audio_mixed.exists():
        _cache_audio_data(str(record_dir), _audio_mtime(record_dir))


# =============================================================================
# Streamlit tab renderer
# =============================================================================


def render_audio_analysis_tab(record_dir: Path) -> None:
    """Render the Audio Analysis tab for a given record / trial directory."""
    st.markdown("### Audio Analysis")

    events_file = record_dir / "elevenlabs_events.jsonl"
    audio_mixed = next(record_dir.glob("audio_mixed*.wav"), record_dir / "audio_mixed.wav")

    if not events_file.exists() and not audio_mixed.exists():
        st.info("No audio files found in this record directory.")
        return

    try:
        # Cache hit when mtime unchanged; re-loads if the file was updated.
        data = _cache_audio_data(str(record_dir), _audio_mtime(record_dir))
    except Exception as exc:
        st.error(f"Could not load audio data: {exc}")
        return

    # Spectrogram toggles — side-by-side when EL is available, single when not
    if data["el_loaded"]:
        col1, col2 = st.columns(2)
        with col1:
            show_mixed_spec = st.checkbox("Show Mixed Audio Spectrogram", value=False)
        with col2:
            show_el_spec = st.checkbox("Show ElevenLabs Spectrogram", value=False)
    else:
        show_mixed_spec = st.checkbox("Show Mixed Audio Spectrogram", value=False)
        show_el_spec = False
        st.info("ElevenLabs audio recording is not available for this record.")

    try:
        fig = _build_figure(data, show_mixed_spec=show_mixed_spec, show_el_spec=show_el_spec)
        st.plotly_chart(fig, width="stretch", theme="streamlit")
    except Exception as exc:
        st.error(f"Could not render audio plot: {exc}")
