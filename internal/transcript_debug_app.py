#!/usr/bin/env python3
"""Streamlit Transcript Debug App for Voice Agent Benchmark

Runs MetricsContextProcessor on all records in a run directory and displays
a side-by-side comparison of conversation_trace against the underlying
per-role variables (transcribed_user_turns, intended_user_turns,
turns_llm_response_assistant, etc.) to surface alignment issues.

Usage:
    streamlit run scripts/transcript_debug_app.py
"""

import io
import json
from pathlib import Path
from typing import Optional

import jiwer
import streamlit as st
from pydub import AudioSegment

from eva.metrics.processor import MetricsContextProcessor, _ProcessorContext
from eva.models.results import ConversationResult
from eva.utils.log_processing import AnnotationLabel
from eva.utils.logging import get_logger
from eva.utils.wer_normalization import normalize_text

logger = get_logger(__name__)

OUTPUT_DIR = Path("output")


# ============================================================================
# Data Loading
# ============================================================================


def get_run_directories() -> list[Path]:
    """Get all run directories in output/, sorted newest first."""
    if not OUTPUT_DIR.exists():
        return []
    run_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and (d / "records").exists()]
    return sorted(run_dirs, key=lambda d: d.name, reverse=True)


def get_record_directories(run_dir: Path) -> list[Path]:
    """Get all leaf record directories in a run, sorted by ID.

    Supports both flat layout (records/<id>/result.json) and trial layout
    (records/<id>/trial_X/result.json).  Returns the directories that
    actually contain result.json.
    """
    records_dir = run_dir / "records"
    if not records_dir.exists():
        return []

    leaf_dirs: list[Path] = []
    for record_dir in sorted(records_dir.iterdir(), key=lambda d: d.name):
        if not record_dir.is_dir():
            continue
        if "failed_attempt" in record_dir.name:
            continue
        # Flat layout: result.json directly in record dir
        if (record_dir / "result.json").exists():
            leaf_dirs.append(record_dir)
        else:
            # Trial layout: look for trial_*/result.json subdirectories
            trial_dirs = sorted(
                [
                    t
                    for t in record_dir.iterdir()
                    if t.is_dir() and (t / "result.json").exists() and "failed_attempt" not in t.name
                ],
                key=lambda d: d.name,
            )
            leaf_dirs.extend(trial_dirs)

    return leaf_dirs


def _display_key(record_dir: Path, records_root: Path) -> str:
    """Build a display key from the record directory path relative to records/.

    For flat layout returns e.g. "1.1.3", for trial layout "1.1.3/trial_0".
    """
    return str(record_dir.relative_to(records_root))


@st.cache_data(show_spinner="Processing all records...")
def load_all_contexts(run_dir_str: str) -> dict[str, Optional[dict]]:
    """Process all records and return serialised _ProcessorContext dicts keyed by display key."""
    run_dir = Path(run_dir_str)
    records_root = run_dir / "records"
    processor = MetricsContextProcessor()
    contexts: dict[str, Optional[dict]] = {}

    for record_dir in get_record_directories(run_dir):
        result_path = record_dir / "result.json"
        if not result_path.exists():
            continue
        key = _display_key(record_dir, records_root)
        try:
            with open(result_path) as f:
                result = ConversationResult(**json.load(f))
            # Use the display key (includes trial name) as record_id for logging
            result.record_id = key
            ctx = processor.process_record(result, record_dir)
            if ctx is not None:
                audio_path = record_dir / "audio_mixed.wav"
                ctx_dict = _context_to_dict(ctx, audio_path)
                ctx_dict["_record_dir"] = str(record_dir)
                contexts[key] = ctx_dict
            else:
                contexts[key] = {"_error": "process_record returned None (check logs for details)"}
        except Exception as e:
            logger.warning(f"Failed to process {record_dir.name}: {e}")
            contexts[key] = {"_error": f"{type(e).__name__}: {e}"}

    return contexts


def _context_to_dict(ctx: _ProcessorContext, audio_path: Optional[Path] = None) -> dict:
    """Serialise a _ProcessorContext into a plain dict for caching."""
    # Build a linear mapping from ElevenLabs audio timestamps to WAV offsets.
    #
    # The ElevenLabs timestamps come from time.time() on the user-simulator
    # process.  The WAV is recorded on the assistant-server process.  The two
    # clocks drift apart so a simple subtraction gives growing error.
    #
    # Anchors:
    #   - session_started (EL) ≈ WAV t=0  (recording starts at session open)
    #   - session_ended   (EL) ≈ WAV end  (recording stops at session close)
    #
    # Linear mapping:
    #   wav_offset = (el_ts - session_started) * (wav_duration / session_duration)

    session_started_sec: Optional[float] = None
    session_ended_sec: Optional[float] = None

    for event in ctx.history:
        if event["source"] != "elevenlabs" or event["event_type"] != "connection_state":
            continue
        data = event.get("data", {})
        inner = data.get("data", {}) if isinstance(data, dict) else {}
        if not isinstance(inner, dict):
            continue
        state = inner.get("state")
        if state == "session_started" and session_started_sec is None:
            session_started_sec = event["timestamp_ms"] / 1000.0
        elif state == "session_ended":
            session_ended_sec = event["timestamp_ms"] / 1000.0

    # Compute scale from WAV duration vs session duration
    wav_duration_sec: Optional[float] = None
    if audio_path and audio_path.exists():
        try:
            audio = AudioSegment.from_wav(str(audio_path))
            wav_duration_sec = len(audio) / 1000.0
        except Exception:
            pass

    clock_scale = 1.0
    if session_started_sec and session_ended_sec and wav_duration_sec:
        session_dur = session_ended_sec - session_started_sec
        if session_dur > 0:
            clock_scale = wav_duration_sec / session_dur

    return {
        "record_id": ctx.record_id,
        "session_started_sec": session_started_sec,
        "clock_scale": clock_scale,
        "conversation_trace": ctx.conversation_trace,
        "transcribed_user_turns": ctx.transcribed_user_turns,
        "transcribed_assistant_turns": ctx.transcribed_assistant_turns,
        "intended_user_turns": ctx.intended_user_turns,
        "intended_assistant_turns": ctx.intended_assistant_turns,
        "audio_timestamps_user_turns": {k: list(v or []) for k, v in ctx.audio_timestamps_user_turns.items()},
        "audio_timestamps_assistant_turns": {k: list(v or []) for k, v in ctx.audio_timestamps_assistant_turns.items()},
        "assistant_interrupted_turns": sorted(ctx.assistant_interrupted_turns),
        "user_interrupted_turns": sorted(ctx.user_interrupted_turns),
        "num_assistant_turns": ctx.num_assistant_turns,
        "num_user_turns": ctx.num_user_turns,
        "num_tool_calls": ctx.num_tool_calls,
        "tool_called": ctx.tool_called,
        "tool_params": ctx.tool_params,
        "tool_responses": ctx.tool_responses,
        "conversation_finished": ctx.conversation_finished,
        "history": ctx.history,
    }


# ============================================================================
# Audio Helpers
# ============================================================================


@st.cache_data(show_spinner=False)
def load_audio(audio_path: str) -> Optional[bytes]:
    """Load a WAV file and return raw bytes (cached)."""
    p = Path(audio_path)
    if not p.exists():
        return None
    return p.read_bytes()


def el_timestamp_to_wav_offset(ctx: dict, el_ts_sec: float) -> float:
    """Convert an ElevenLabs absolute timestamp to a WAV-file offset in seconds.

    Uses a linear mapping:
        wav_offset = (el_ts - session_started) * clock_scale
    where clock_scale = wav_duration / session_duration corrects for clock
    drift between the user-simulator and assistant-server processes.
    """
    session_start = ctx.get("session_started_sec", 0.0)
    scale = ctx.get("clock_scale", 1.0)
    return (el_ts_sec - session_start) * scale


def extract_audio_segment(audio_bytes: bytes, start_sec: float, end_sec: float, padding_sec: float = 0.5) -> bytes:
    """Extract a segment from a WAV file, return as WAV bytes.

    Args:
        audio_bytes: Full WAV file bytes.
        start_sec: Start offset in seconds from beginning of file.
        end_sec: End offset in seconds from beginning of file.
        padding_sec: Extra padding before/after the segment.
    """
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    start_ms = max(0, int((start_sec - padding_sec) * 1000))
    end_ms = min(len(audio), int((end_sec + padding_sec) * 1000))
    segment = audio[start_ms:end_ms]
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return buf.getvalue()


def render_segment_player(
    audio_bytes: Optional[bytes],
    ctx: dict,
    timestamps: list[float],
    label: str,
    key: str,
):
    """Render an inline audio player for a turn segment.

    Args:
        audio_bytes: Full WAV bytes (or None if unavailable).
        ctx: Serialised context dict (contains clock mapping params).
        timestamps: [start, end] absolute ElevenLabs timestamps (seconds).
        label: Display label (e.g. "User" or "Assistant").
        key: Unique streamlit widget key.
    """
    if audio_bytes is None or not timestamps or ctx.get("session_started_sec") is None:
        return

    start_offset = el_timestamp_to_wav_offset(ctx, timestamps[0])
    end_offset = el_timestamp_to_wav_offset(ctx, timestamps[1])
    duration = end_offset - start_offset

    if st.button(f"Play {label} ({start_offset:.1f}s - {end_offset:.1f}s, {duration:.1f}s)", key=key):
        segment = extract_audio_segment(audio_bytes, start_offset, end_offset)
        st.audio(segment, format="audio/wav")


# ============================================================================
# WER Comparison
# ============================================================================


def _compute_turn_wer(
    tts_text: dict,
    transcript: dict,
    language: str = "en",
) -> dict[int, dict]:
    """Compute per-turn WER between TTS source text (reference) and STT transcript (hypothesis).

    Args:
        tts_text: Dict of turn_id -> text that TTS was asked to speak (reference).
        transcript: Dict of turn_id -> text that STT transcribed (hypothesis).
        language: Language code for normalization.

    Returns a dict keyed by turn_id with:
        {wer, ref, hyp, ref_clean, hyp_clean, errors}
    """
    results: dict[int, dict] = {}
    common_keys = set(tts_text.keys()) & set(transcript.keys())

    for turn_id in sorted(common_keys):
        ref = tts_text[turn_id]
        hyp = transcript[turn_id]

        if not ref or not hyp:
            continue

        ref_clean = normalize_text(ref, language)
        hyp_clean = normalize_text(hyp, language)

        if not ref_clean.strip() or not hyp_clean.strip():
            continue

        turn_wer = jiwer.wer(ref_clean, hyp_clean)
        output = jiwer.process_words(ref_clean, hyp_clean)

        # Extract error details
        errors: list[dict] = []
        ref_words = output.references[0]
        hyp_words = output.hypotheses[0]
        if not isinstance(ref_words, list):
            ref_words = ref_words.split()
        if not isinstance(hyp_words, list):
            hyp_words = hyp_words.split()

        for chunk in output.alignments[0]:
            if chunk.type == "substitute":
                errors.append(
                    {
                        "type": "sub",
                        "expected": " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx]),
                        "actual": " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]),
                    }
                )
            elif chunk.type == "delete":
                errors.append(
                    {
                        "type": "del",
                        "word": " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx]),
                    }
                )
            elif chunk.type == "insert":
                errors.append(
                    {
                        "type": "ins",
                        "word": " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]),
                    }
                )

        results[turn_id] = {
            "wer": round(turn_wer, 3),
            "ref": ref,
            "hyp": hyp,
            "ref_clean": ref_clean,
            "hyp_clean": hyp_clean,
            "errors": errors,
        }

    return results


def compute_user_turn_wer(ctx: dict, language: str = "en") -> dict[int, dict]:
    """Compute per-turn WER for user turns (tts_text_user vs transcript_user)."""
    return _compute_turn_wer(
        ctx.get("intended_user_turns", {}),
        ctx.get("transcribed_user_turns", {}),
        language,
    )


def compute_assistant_turn_wer(ctx: dict, language: str = "en") -> dict[int, dict]:
    """Compute per-turn WER for assistant turns (tts_text_assistant vs transcript_assistant)."""
    return _compute_turn_wer(
        ctx.get("intended_assistant_turns", {}),
        ctx.get("transcribed_assistant_turns", {}),
        language,
    )


def _wer_badge(wer: float) -> str:
    """Return a colored WER badge string."""
    if wer == 0.0:
        return ":green[WER 0.0]"
    elif wer <= 0.1:
        return f":green[WER {wer:.2f}]"
    elif wer <= 0.3:
        return f":orange[WER {wer:.2f}]"
    else:
        return f":red[WER {wer:.2f}]"


def _wer_color(wer: float) -> str:
    """Return a hex color for the WER value."""
    if wer == 0.0:
        return "#4caf50"
    elif wer <= 0.1:
        return "#66bb6a"
    elif wer <= 0.3:
        return "#ff9800"
    else:
        return "#f44336"


# ============================================================================
# Issue Detection
# ============================================================================


def detect_issues(ctx: dict) -> list[dict]:
    """Detect alignment and data issues in a processed context.

    Returns a list of {severity, category, message} dicts.
    """
    if _is_failed(ctx):
        return []
    issues: list[dict] = []

    user_transcript_keys = set(ctx.get("transcribed_user_turns", {}).keys())
    user_tts_keys = set(ctx.get("intended_user_turns", {}).keys())
    asst_transcript_keys = set(ctx.get("transcribed_assistant_turns", {}).keys())
    asst_tts_keys = set(ctx.get("intended_assistant_turns", {}).keys())
    user_audio_keys = set(ctx.get("audio_timestamps_user_turns", {}).keys())
    asst_audio_keys = set(ctx.get("audio_timestamps_assistant_turns", {}).keys())

    # 1. User turn-id alignment: transcript vs tts_text
    only_transcript = user_transcript_keys - user_tts_keys
    only_tts = user_tts_keys - user_transcript_keys
    if only_transcript:
        issues.append(
            {
                "severity": "warning",
                "category": "User turn alignment",
                "message": f"transcribed_user_turns has turn(s) {sorted(only_transcript)} missing from intended_user_turns",
            }
        )
    if only_tts:
        issues.append(
            {
                "severity": "warning",
                "category": "User turn alignment",
                "message": f"intended_user_turns has turn(s) {sorted(only_tts)} missing from transcribed_user_turns",
            }
        )

    # 2. Assistant turn-id alignment: tts_text vs transcript vs llm_response
    only_tts_asst = asst_tts_keys - asst_transcript_keys
    only_transcript_asst = asst_transcript_keys - asst_tts_keys
    if only_tts_asst:
        issues.append(
            {
                "severity": "warning",
                "category": "Assistant turn alignment",
                "message": f"intended_assistant_turns has turn(s) {sorted(only_tts_asst)} missing from transcribed_assistant_turns",
            }
        )
    if only_transcript_asst:
        issues.append(
            {
                "severity": "warning",
                "category": "Assistant turn alignment",
                "message": f"transcribed_assistant_turns has turn(s) {sorted(only_transcript_asst)} missing from intended_assistant_turns",
            }
        )

    # 3. Audio timestamp coverage
    missing_asst_audio = asst_tts_keys - asst_audio_keys
    if missing_asst_audio:
        issues.append(
            {
                "severity": "warning",
                "category": "Audio timestamps",
                "message": f"Missing assistant audio timestamps for turn(s) {sorted(missing_asst_audio)}",
            }
        )
    missing_user_audio = user_tts_keys - user_audio_keys
    if missing_user_audio:
        issues.append(
            {
                "severity": "warning",
                "category": "Audio timestamps",
                "message": f"Missing user audio timestamps for turn(s) {sorted(missing_user_audio)}",
            }
        )

    _interruption_labels = (
        AnnotationLabel.ASSISTANT_INTERRUPTS,
        AnnotationLabel.USER_INTERRUPTS,
        AnnotationLabel.CUT_OFF_BY_USER,
        AnnotationLabel.CUT_OFF_BY_ASSISTANT,
        AnnotationLabel.CUT_OFF_ON_ITS_OWN,
    )
    for entry in ctx["conversation_trace"]:
        content = entry.get("content", "")
        if isinstance(content, str) and any(label in content for label in _interruption_labels):
            issues.append(
                {
                    "severity": "info",
                    "category": "Interruption",
                    "message": f"Turn {entry.get('turn_id')}: {content[:120]}...",
                }
            )

    # 5. Turn-id gaps
    if user_transcript_keys:
        expected = set(range(min(user_transcript_keys), max(user_transcript_keys) + 1))
        gaps = sorted(expected - user_transcript_keys)
        if gaps:
            issues.append(
                {
                    "severity": "warning",
                    "category": "Turn gaps",
                    "message": f"transcribed_user_turns has gaps at turn(s) {gaps}",
                }
            )

    # 7. High user STT WER (transcript vs tts_text)
    wer_results = compute_user_turn_wer(ctx)
    high_wer_turns = [(tid, r["wer"]) for tid, r in wer_results.items() if r["wer"] > 0.5]
    if high_wer_turns:
        turn_strs = [f"{tid} ({wer:.0%})" for tid, wer in high_wer_turns]
        issues.append(
            {
                "severity": "warning",
                "category": "User STT WER",
                "message": f"High WER (>50%) on user turn(s): {', '.join(turn_strs)}",
            }
        )

    # 8. High assistant STT WER (transcript vs tts_text)
    asst_wer_results = compute_assistant_turn_wer(ctx)
    high_asst_wer = [(tid, r["wer"]) for tid, r in asst_wer_results.items() if r["wer"] > 0.5]
    if high_asst_wer:
        turn_strs = [f"{tid} ({wer:.0%})" for tid, wer in high_asst_wer]
        issues.append(
            {
                "severity": "warning",
                "category": "Assistant STT WER",
                "message": f"High WER (>50%) on assistant turn(s): {', '.join(turn_strs)}",
            }
        )

    return issues


# ============================================================================
# Rendering
# ============================================================================

# Shared CSS injected once at render time
_CSS = """
<style>
.turn-header {
    font-weight: 600; font-size: 0.95em; margin: 16px 0 4px 0;
    padding: 4px 0; border-bottom: 1px solid #444;
}
.bubble {
    padding: 8px 12px; border-radius: 8px; margin: 3px 0;
    font-size: 0.9em; line-height: 1.45;
}
.bubble-user     { background: #1a237e; color: #bbdefb; }
.bubble-asst     { background: #1b5e20; color: #c8e6c9; }
.bubble-asst-int { background: #b71c1c; color: #ffcdd2; }
.bubble-tool     { background: #263238; color: #b0bec5; font-family: monospace; font-size: 0.82em; }
.bubble-tool-err { background: #3e2723; color: #b0bec5; font-family: monospace; font-size: 0.82em; }
.cell {
    padding: 6px 10px; border-radius: 6px; margin: 3px 0;
    font-size: 0.85em; line-height: 1.4;
}
.cell-user { background: #0d1b3e; color: #90caf9; }
.cell-asst { background: #0d2b0d; color: #a5d6a7; }
.cell-empty { color: #666; font-style: italic; font-size: 0.82em; }
.cell-label { font-size: 0.72em; color: #888; margin-bottom: 2px; }
</style>
"""


def _get_val(d: dict, turn_id: int) -> Optional[str]:
    """Look up turn_id as int or str key."""
    return d.get(turn_id) or d.get(str(turn_id))


def _build_turn_groups(ctx: dict) -> list[dict]:
    """Group conversation_trace entries by turn_id, preserving order.

    Includes turn IDs from all per-turn variables so that turns present in
    transcript/tts_text but missing from conversation_trace still
    appear in the view.

    Returns a list of dicts:
        {turn_id, entries: [...], has_user, has_assistant}
    """
    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    for entry in ctx["conversation_trace"]:
        tid = entry.get("turn_id")
        if tid is None:
            continue
        if tid not in groups:
            groups[tid] = []
            order.append(tid)
        groups[tid].append(entry)

    # Collect all turn IDs from per-turn variables that may not be in
    # conversation_trace.
    all_turn_ids: set[int] = set(groups.keys())
    for var_name in (
        "transcribed_user_turns",
        "transcribed_assistant_turns",
        "intended_user_turns",
        "intended_assistant_turns",
        "audio_timestamps_user_turns",
        "audio_timestamps_assistant_turns",
    ):
        for k in ctx.get(var_name, {}):
            all_turn_ids.add(int(k))

    # Add any missing turn IDs
    for tid in sorted(all_turn_ids - set(order)):
        groups[tid] = []
    order = sorted(all_turn_ids)

    result = []
    for tid in order:
        entries = groups[tid]
        has_user = any(e.get("role") == "user" for e in entries)
        has_assistant = any(e.get("role") == "assistant" for e in entries)
        # Also check per-turn variables so columns 2/3 render even when
        # conversation_trace has no entry for this turn.
        if not has_user:
            has_user = bool(
                _get_val(ctx.get("transcribed_user_turns", {}), tid)
                or _get_val(ctx.get("intended_user_turns", {}), tid)
            )
        if not has_assistant:
            has_assistant = bool(
                _get_val(ctx.get("transcribed_assistant_turns", {}), tid)
                or _get_val(ctx.get("intended_assistant_turns", {}), tid)
            )
        result.append(
            {
                "turn_id": tid,
                "entries": entries,
                "has_user": has_user,
                "has_assistant": has_assistant,
            }
        )
    return result


def _render_entry_html(entry: dict) -> str:
    """Render a single conversation_trace entry as an HTML bubble."""
    role = entry.get("role", "")
    content = entry.get("content", "")
    entry_type = entry.get("type", "")

    _int_markers = (
        AnnotationLabel.ASSISTANT_INTERRUPTS,
        AnnotationLabel.USER_INTERRUPTS,
        AnnotationLabel.CUT_OFF_BY_USER,
        AnnotationLabel.CUT_OFF_BY_ASSISTANT,
        AnnotationLabel.CUT_OFF_ON_ITS_OWN,
    )
    is_interruption = isinstance(content, str) and any(m in content for m in _int_markers)

    if role == "user":
        cls = "bubble-asst-int" if is_interruption else "bubble-user"
        return f'<div class="bubble {cls}"><small style="opacity:0.5">{entry_type}</small><br>{content}</div>'
    elif role == "assistant":
        cls = "bubble-asst-int" if is_interruption else "bubble-asst"
        return f'<div class="bubble {cls}"><small style="opacity:0.5">{entry_type}</small><br>{content}</div>'
    elif entry_type == "tool_call":
        tool = entry.get("tool_name", "?")
        params = json.dumps(entry.get("parameters", {}))
        return f'<div class="bubble bubble-tool"><b>TOOL CALL</b>: {tool}({params})</div>'
    elif entry_type == "tool_response":
        resp = entry.get("tool_response", "")
        resp_str = json.dumps(resp) if isinstance(resp, dict) else str(resp)
        is_error = isinstance(resp, dict) and resp.get("is_error")
        cls = "bubble-tool-err" if is_error else "bubble-tool"
        return f'<div class="bubble {cls}"><b>TOOL RESPONSE</b>: {resp_str}</div>'
    return ""


def _render_cell_html(value: Optional[str], css_class: str) -> str:
    """Render a transcript/tts_text value as a styled div."""
    if not value:
        return '<div class="cell-empty">--</div>'
    return f'<div class="cell {css_class}">{value}</div>'


def render_record(
    ctx: dict,
    audio_bytes: Optional[bytes] = None,
):
    """Render the combined 3-column view: transcript_with_tools | transcribed | intended."""
    st.markdown(_CSS, unsafe_allow_html=True)

    # Pre-compute per-turn WER
    wer_results = compute_user_turn_wer(ctx)
    asst_wer_results = compute_assistant_turn_wer(ctx)

    # Column headers
    hdr1, hdr2, hdr3 = st.columns(3)
    hdr1.markdown("**conversation_trace**")
    hdr2.markdown("**turns_transcript** (STT)")
    hdr3.markdown("**intended_turns** (TTS source)")
    st.divider()

    turn_groups = _build_turn_groups(ctx)

    for group in turn_groups:
        tid = group["turn_id"]
        entries = group["entries"]

        # --- Column 1: transcript_with_tools entries ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'<div class="turn-header">Turn {tid}</div>', unsafe_allow_html=True)
            html = "".join(_render_entry_html(e) for e in entries)
            st.markdown(html, unsafe_allow_html=True)

        # --- Column 2: turns_transcript (user=pipecat STT, assistant=ElevenLabs STT) ---
        with col2:
            wer_badges = []
            user_wer_info = wer_results.get(tid)
            asst_wer_info = asst_wer_results.get(tid)
            if user_wer_info is not None:
                color = _wer_color(user_wer_info["wer"])
                wer_badges.append(
                    f'<span style="color:{color};font-weight:600;font-size:0.8em">U:{user_wer_info["wer"]:.0%}</span>'
                )
            if asst_wer_info is not None:
                color = _wer_color(asst_wer_info["wer"])
                wer_badges.append(
                    f'<span style="color:{color};font-weight:600;font-size:0.8em">A:{asst_wer_info["wer"]:.0%}</span>'
                )
            wer_label = f" {' '.join(wer_badges)}" if wer_badges else ""
            st.markdown(f'<div class="turn-header">Turn {tid}{wer_label}</div>', unsafe_allow_html=True)
            parts = []
            if group["has_user"]:
                val = _get_val(ctx["transcribed_user_turns"], tid)
                parts.append('<div class="cell-label">user (pipecat STT)</div>')
                parts.append(_render_cell_html(val, "cell-user"))
            if group["has_assistant"]:
                val = _get_val(ctx["transcribed_assistant_turns"], tid)
                parts.append('<div class="cell-label">assistant (ElevenLabs STT)</div>')
                parts.append(_render_cell_html(val, "cell-asst"))
            st.markdown("".join(parts), unsafe_allow_html=True)

        # --- Column 3: intended_turns (user=ElevenLabs, assistant=pipecat) ---
        with col3:
            st.markdown(f'<div class="turn-header">Turn {tid}</div>', unsafe_allow_html=True)
            parts = []
            if group["has_user"]:
                val = _get_val(ctx["intended_user_turns"], tid)
                parts.append('<div class="cell-label">user (ElevenLabs TTS)</div>')
                parts.append(_render_cell_html(val, "cell-user"))
            if group["has_assistant"]:
                val = _get_val(ctx["intended_assistant_turns"], tid)
                parts.append('<div class="cell-label">assistant (pipecat TTS)</div>')
                parts.append(_render_cell_html(val, "cell-asst"))
            st.markdown("".join(parts), unsafe_allow_html=True)

        # --- WER error details (shown below columns when errors exist) ---
        user_wer_info = wer_results.get(tid)
        asst_wer_info = asst_wer_results.get(tid)
        if user_wer_info and user_wer_info["wer"] > 0.0 and user_wer_info["errors"]:
            error_parts = []
            for e in user_wer_info["errors"]:
                if e["type"] == "sub":
                    error_parts.append(f"~~{e['expected']}~~ → **{e['actual']}**")
                elif e["type"] == "del":
                    error_parts.append(f"~~{e['word']}~~ (deleted)")
                elif e["type"] == "ins":
                    error_parts.append(f"**+{e['word']}** (inserted)")
            if error_parts:
                st.caption(f"User STT errors: {' | '.join(error_parts)}")
        if asst_wer_info and asst_wer_info["wer"] > 0.0 and asst_wer_info["errors"]:
            error_parts = []
            for e in asst_wer_info["errors"]:
                if e["type"] == "sub":
                    error_parts.append(f"~~{e['expected']}~~ → **{e['actual']}**")
                elif e["type"] == "del":
                    error_parts.append(f"~~{e['word']}~~ (deleted)")
                elif e["type"] == "ins":
                    error_parts.append(f"**+{e['word']}** (inserted)")
            if error_parts:
                st.caption(f"Assistant STT errors: {' | '.join(error_parts)}")

        # --- Audio segment players below the 3 columns ---
        user_ts = _get_val(ctx["audio_timestamps_user_turns"], tid)
        asst_ts = _get_val(ctx["audio_timestamps_assistant_turns"], tid)

        all_timestamps = {**ctx["audio_timestamps_user_turns"], **ctx["audio_timestamps_assistant_turns"]}
        all_starts = [segs[0][0] for segs in all_timestamps.values() if segs]
        t0 = min(all_starts) if all_starts else 0

        st.markdown(f"User timestamps {user_ts}")
        st.markdown(f"Assistant timestamps {asst_ts}")

        if user_ts and asst_ts:
            all_segs = sorted(
                [("user", s - t0, e - t0) for s, e in user_ts] + [("assistant", s - t0, e - t0) for s, e in asst_ts],
                key=lambda x: x[1],
            )
            seg_latencies = []
            for i in range(len(all_segs) - 1):
                prev_role, _, prev_end = all_segs[i]
                next_role, next_start, _ = all_segs[i + 1]
                if prev_role != next_role:
                    gap = next_start - prev_end
                    seg_latencies.append(f"{prev_role}_end→{next_role}_start: {gap:.3f}s")

            st.markdown(f"Latency: {seg_latencies}")
        # if user_ts or asst_ts:
        #     acol1, acol2 = st.columns(2)
        #     with acol1:
        #         if user_ts:
        #             render_segment_player(audio_bytes, ctx, user_ts, "User", key=f"seg_u_{tid}")
        #     with acol2:
        #         if asst_ts:
        #             render_segment_player(audio_bytes, ctx, asst_ts, "Assistant", key=f"seg_a_{tid}")

        st.divider()


def render_issues_summary(issues: list[dict]):
    """Render the issue list with severity badges."""
    if not issues:
        st.success("No issues detected")
        return

    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]

    if warnings:
        st.warning(f"{len(warnings)} warning(s)")
    if infos:
        st.info(f"{len(infos)} informational note(s)")

    for issue in issues:
        icon = {
            "warning": ":orange[WARNING]",
            "info": ":blue[INFO]",
            "error": ":red[ERROR]",
        }.get(issue["severity"], issue["severity"])
        st.markdown(f"- {icon} **{issue['category']}**: {issue['message']}")


def _is_failed(ctx: Optional[dict]) -> bool:
    """Return True if the context represents a failed record."""
    return ctx is None or "_error" in ctx


def render_run_overview(all_contexts: dict[str, Optional[dict]]):
    """Render an overview table of all records with issue counts."""
    rows = []
    for record_id, ctx in sorted(all_contexts.items()):
        if _is_failed(ctx):
            error_msg = ctx.get("_error", "Unknown error") if ctx else "Unknown error"
            rows.append(
                {
                    "Record": record_id,
                    "Status": "FAILED",
                    "Error": error_msg,
                    "User Turns": None,
                    "Asst Turns": None,
                    "Tool Calls": None,
                    "User WER": None,
                    "Asst WER": None,
                    "Interruptions": None,
                    "Warning Categories": "",
                }
            )
            continue
        issues = detect_issues(ctx)
        interruptions = sum(1 for i in issues if i["category"] == "Interruption")
        # Collect unique warning categories
        warning_categories = sorted({i["category"] for i in issues if i["severity"] == "warning"})
        user_wer = compute_user_turn_wer(ctx)
        asst_wer = compute_assistant_turn_wer(ctx)
        avg_user_wer = None
        if user_wer:
            vals = [r["wer"] for r in user_wer.values()]
            avg_user_wer = round(sum(vals) / len(vals), 3)
        avg_asst_wer = None
        if asst_wer:
            vals = [r["wer"] for r in asst_wer.values()]
            avg_asst_wer = round(sum(vals) / len(vals), 3)
        rows.append(
            {
                "Record": record_id,
                "Status": "OK",
                "Error": "",
                "User Turns": ctx["num_user_turns"],
                "Asst Turns": ctx["num_assistant_turns"],
                "Tool Calls": ctx["num_tool_calls"],
                "User WER": avg_user_wer,
                "Asst WER": avg_asst_wer,
                "Interruptions": interruptions,
                "Warning Categories": ", ".join(warning_categories),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)


# ============================================================================
# Main App
# ============================================================================


def main():
    st.set_page_config(
        page_title="Transcript Debug",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Transcript Debug Inspector")

    # --- Sidebar: run and record selection ---
    run_dirs = get_run_directories()
    if not run_dirs:
        st.error(f"No run directories found in {OUTPUT_DIR}")
        return

    st.sidebar.header("Run Selection")
    run_dir_names = [d.name for d in run_dirs]
    qp = st.query_params

    # Resolve initial run index from query param
    run_index = 0
    if "run" in qp and qp["run"] in run_dir_names:
        run_index = run_dir_names.index(qp["run"])
    selected_run_name = st.sidebar.selectbox("Select Run", run_dir_names, index=run_index)
    selected_run_dir = OUTPUT_DIR / selected_run_name

    # Process all records
    all_contexts = load_all_contexts(str(selected_run_dir))

    if not all_contexts:
        st.error("No records could be processed")
        return

    # Record selector
    st.sidebar.header("Record Selection")
    valid_records = sorted(all_contexts.keys())

    # Resolve initial record index from query param
    record_index = 0
    if "record" in qp and qp["record"] in valid_records:
        record_index = valid_records.index(qp["record"])
    selected_record = st.sidebar.selectbox("Select Record", valid_records, index=record_index)

    # Sync query params
    st.query_params["run"] = selected_run_name
    st.query_params["record"] = selected_record

    # Issue filter
    st.sidebar.header("Filters")
    only_with_issues = st.sidebar.checkbox("Only records with warnings", value=False)

    if only_with_issues:
        filtered = {
            rid: ctx
            for rid, ctx in all_contexts.items()
            if _is_failed(ctx) or any(i["severity"] == "warning" for i in detect_issues(ctx))
        }
        st.sidebar.caption(f"{len(filtered)} / {len(all_contexts)} records with warnings")
    else:
        filtered = all_contexts

    # --- Main content ---
    tab_overview, tab_detail = st.tabs(["Overview", "Record Detail"])

    with tab_overview:
        st.markdown("### Run Overview")
        render_run_overview(filtered)

    with tab_detail:
        ctx = all_contexts.get(selected_record)
        if _is_failed(ctx):
            error_msg = ctx.get("_error", "Unknown error") if ctx else "Unknown error"
            st.error(f"Record {selected_record} failed to process: {error_msg}")
            return

        # Load audio for segment playback
        record_dir = Path(ctx.get("_record_dir", selected_run_dir / "records" / selected_record))
        audio_path = record_dir / "audio_mixed.wav"
        audio_bytes = load_audio(str(audio_path))

        # Full recording player
        if audio_bytes:
            st.markdown("**Full recording**")
            st.audio(audio_bytes, format="audio/wav")

        # Issues
        issues = detect_issues(ctx)
        with st.expander(f"Issues ({len(issues)})", expanded=bool(issues)):
            render_issues_summary(issues)

        # Raw data expanders
        with st.expander("Raw: history"):
            st.json(ctx.get("history", []))

        raw_vars = [
            "transcribed_assistant_turns",
            "intended_assistant_turns",
            "audio_timestamps_assistant_turns",
            "transcribed_user_turns",
            "intended_user_turns",
            "audio_timestamps_user_turns",
        ]
        with st.expander("Raw: per-turn variables"):
            for var_name in raw_vars:
                st.markdown(f"**{var_name}**")
                data = ctx.get(var_name, {})
                # Show sorted by turn id
                st.json({str(k): v for k, v in sorted(data.items())})

        # User STT WER summary
        user_wer = compute_user_turn_wer(ctx)
        if user_wer:
            wer_values = [r["wer"] for r in user_wer.values()]
            avg_wer = sum(wer_values) / len(wer_values)
            with st.expander(
                f"User STT WER ({avg_wer:.0%} avg across {len(wer_values)} turns)",
                expanded=avg_wer > 0.15,
            ):
                wer_cols = st.columns(min(len(user_wer), 6))
                for i, (tid, info) in enumerate(sorted(user_wer.items())):
                    col = wer_cols[i % len(wer_cols)]
                    col.metric(
                        label=f"Turn {tid}",
                        value=f"{info['wer']:.0%}",
                    )

        # Assistant STT WER summary
        asst_wer = compute_assistant_turn_wer(ctx)
        if asst_wer:
            wer_values = [r["wer"] for r in asst_wer.values()]
            avg_wer = sum(wer_values) / len(wer_values)
            with st.expander(
                f"Assistant STT WER ({avg_wer:.0%} avg across {len(wer_values)} turns)",
                expanded=avg_wer > 0.15,
            ):
                wer_cols = st.columns(min(len(asst_wer), 6))
                for i, (tid, info) in enumerate(sorted(asst_wer.items())):
                    col = wer_cols[i % len(wer_cols)]
                    col.metric(
                        label=f"Turn {tid}",
                        value=f"{info['wer']:.0%}",
                    )

        st.divider()

        render_record(ctx, audio_bytes)


if __name__ == "__main__":
    main()
