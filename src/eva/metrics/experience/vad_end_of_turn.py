"""VAD-analyzer turn diagnostics for the turn_taking metric.

Surfaces Krisp's "no silence-duration fallback" hang and Smart Turn's silence-fallback
(stop_secs) usage directly in turn_taking.sub_metrics, for any pipecat run with local
turn-analyzer VAD (Krisp or Smart Turn). Null for runs with no local VAD (other
frameworks, or external turn strategies like self-endpointing STT/FLUX).

No pipecat modifications: all signals come from files eva already writes
(pipecat_logs.jsonl, pipecat_metrics.jsonl) or pipecat's own log output (logs.log).
"""

import json
import re
import statistics
from pathlib import Path
from typing import Any

from eva.metrics.base import MetricContext
from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.models.results import MetricScore
from eva.utils.conversation_correctly_finished.final_turn import final_turn_input_flags
from eva.utils.json_utils import load_jsonl

_STOP_SECS_RE = re.compile(r"End of Turn complete due to stop_secs\. Silence in ms: ([\d.]+)")
_FINAL_TURN_FLAG_KEYS = ("short", "acknowledgement", "spelled_entity")


def _find_run_config(output_dir: str) -> dict[str, Any] | None:
    """Walk up from output_dir to the run root and load config.json, or None if absent/unreadable."""
    if not output_dir:
        return None
    path = Path(output_dir)
    while True:
        candidate = path / "config.json"
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                return None
        if path == path.parent:
            return None
        path = path.parent


def vad_turn_metrics_applicable(output_dir: str) -> bool:
    """True when this record's run used local turn-analyzer VAD (Krisp or Smart Turn).

    False for non-pipecat frameworks (openai_realtime, elevenlabs, grok_voice,
    smallest_hydra) and for pipecat runs with turn_stop_strategy == "external"
    (self-endpointing STT / external turn detection, e.g. Deepgram FLUX). Also False
    when config.json can't be found at all — a safe default rather than assuming a
    pipeline shape we can't confirm.
    """
    config = _find_run_config(output_dir)
    if config is None:
        return False
    if config.get("framework", "pipecat") != "pipecat":
        return False
    model_cfg = config.get("model") or {}
    return model_cfg.get("turn_stop_strategy", "turn_analyzer") != "external"


def _load_turn_metrics_events(output_dir: str) -> list[dict[str, Any]]:
    """Return TurnMetricsData entries from pipecat_metrics.jsonl, sorted by timestamp.

    Covers both KrispVivaTurn (always is_complete=True when present) and BaseSmartTurn
    (is_complete True or False, one entry per VAD-stop evaluation) — same shape for both.
    """
    entries = load_jsonl(Path(output_dir) / "pipecat_metrics.jsonl")
    events = [
        {
            "timestamp": e["timestamp"],
            "is_complete": e["value"]["is_complete"],
            "e2e_processing_time_ms": e["value"].get("e2e_processing_time_ms"),
        }
        for e in entries
        if e.get("type") == "TurnMetricsData"
    ]
    return sorted(events, key=lambda e: e["timestamp"])


def _load_vad_events(output_dir: str) -> tuple[list[int], list[int]]:
    """Return (sorted user_started_speaking timestamps, sorted user_stopped_speaking timestamps).

    Both epoch-ms, from the same WallClock instance that produces pipecat_metrics.jsonl's
    timestamps (confirmed shared at pipecat_server.py:483-488) — directly comparable.
    """
    entries = load_jsonl(Path(output_dir) / "pipecat_logs.jsonl")
    starts = sorted(e["timestamp"] for e in entries if e.get("type") == "user_started_speaking")
    stops = sorted(e["timestamp"] for e in entries if e.get("type") == "user_stopped_speaking")
    return starts, stops


def _load_transcripts(output_dir: str) -> list[tuple[int, str]]:
    """Return (timestamp, text) for each STT "transcript" event in pipecat_logs.jsonl, sorted.

    Same shared epoch-ms clock as ``_load_vad_events`` (both come from pipecat_logs.jsonl), so a
    transcript's timestamp can be compared directly against a turn window's bounds.
    """
    entries = load_jsonl(Path(output_dir) / "pipecat_logs.jsonl")
    transcripts = [
        (e["timestamp"], e["data"]["frame"])
        for e in entries
        if e.get("type") == "transcript" and e.get("data", {}).get("frame")
    ]
    return sorted(transcripts, key=lambda t: t[0])


def _final_transcript_in_window(
    transcripts: list[tuple[int, str]], window_start: int, window_end: int | None
) -> str | None:
    """Return the last transcript text whose timestamp falls in [window_start, window_end)."""
    in_window = [text for ts, text in transcripts if ts >= window_start and (window_end is None or ts < window_end)]
    return in_window[-1] if in_window else None


def _load_stop_secs_silence_ms(output_dir: str) -> list[float]:
    """Return the 'Silence in ms' value from each pipecat stop_secs forced-completion line, in file order.

    logs.log's local-time text timestamps are not safe to convert and compare against
    pipecat_logs.jsonl/pipecat_metrics.jsonl's shared epoch (confirmed ~1-3s of slop,
    machine/timezone-dependent) — only the count and self-contained silence value are used.
    """
    log_path = Path(output_dir) / "logs.log"
    if not log_path.exists():
        return []
    return [float(m) for m in _STOP_SECS_RE.findall(log_path.read_text(errors="ignore"))]


def _build_turn_windows(vad_starts: list[int]) -> list[tuple[int, int | None]]:
    """Pair consecutive VAD-start timestamps into (window_start, window_end); last window is open-ended."""
    return [(start, vad_starts[i + 1] if i + 1 < len(vad_starts) else None) for i, start in enumerate(vad_starts)]


def _classify_turns(
    vad_starts: list[int],
    vad_stops: list[int],
    turn_metrics_events: list[dict[str, Any]],
    stop_secs_silence_ms: list[float],
    transcripts: list[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Classify each VAD-analyzer turn window as natural/forced/stuck.

    natural: a TurnMetricsData(is_complete=True) entry falls inside the window.
    forced: no natural completion, AND this window has at least one user_stopped_speaking
        event in its range (Smart Turn's stop_secs mechanism only ever fires after VAD has
        registered a stop, so a window with no vad_stop cannot legitimately be the source of
        a forced completion). Such a window consumes the next unused stop_secs silence value
        in file order (Smart Turn only ever has one open turn waiting on stop_secs at a time,
        so ordinal matching is safe among windows that actually have a vad_stop — see spec's
        "Matching stop_secs lines to turn windows"). Close time = last vad_stop in the
        window + that silence value. Also tagged with ``final_turn_flags`` (short /
        acknowledgement / spelled_entity, via ``final_turn_input_flags``) computed on the STT
        transcript matched to this window by timestamp — a candidate explanation for *why* the
        turn analyzer needed the stop_secs fallback instead of closing naturally.
    stuck: neither signal found for this window (including: no natural completion and no
        vad_stop in range, in which case no stop_secs value is consumed — it's left for a
        later window that actually earned it). This is common, not a rare edge
        case (confirmed against real Krisp data: VAD false-starts and silently-swallowed
        utterances both produce it) - a middle window's open_duration_ms uses its own
        window_end (the next VAD start bounds it), never the conversation's global last
        timestamp. Only the true last, open-ended window (window_end is None) falls back
        to last_epoch as a conversation-end proxy.
    """
    windows = _build_turn_windows(vad_starts)
    last_epoch = max([*vad_starts, *vad_stops, *(e["timestamp"] for e in turn_metrics_events)], default=0)
    stop_secs_iter = iter(stop_secs_silence_ms)
    transcripts = transcripts or []
    results: list[dict[str, Any]] = []

    for idx, (window_start, window_end) in enumerate(windows):
        in_window = [
            e
            for e in turn_metrics_events
            if e["timestamp"] >= window_start and (window_end is None or e["timestamp"] < window_end)
        ]
        natural = next((e for e in in_window if e["is_complete"]), None)
        if natural is not None:
            results.append(
                {
                    "turn_index": idx,
                    "open_duration_ms": natural["timestamp"] - window_start,
                    "time_to_complete_ms": natural["e2e_processing_time_ms"],
                    "completion": "natural",
                }
            )
            continue

        stops_in_window = [s for s in vad_stops if s >= window_start and (window_end is None or s < window_end)]
        if stops_in_window:
            silence_ms = next(stop_secs_iter, None)
            if silence_ms is not None:
                last_stop = max(stops_in_window)
                close_time = last_stop + silence_ms
                transcript_text = _final_transcript_in_window(transcripts, window_start, window_end)
                result = {
                    "turn_index": idx,
                    "open_duration_ms": close_time - window_start,
                    "time_to_complete_ms": None,
                    "completion": "forced",
                }
                if transcript_text is not None:
                    # Was stop_secs forced because the user's actual utterance is a shape the
                    # turn analyzer struggles to close on its own (short / ack / spelled-out)?
                    result["transcript_text"] = transcript_text
                    result["final_turn_flags"] = final_turn_input_flags(transcript_text)
                results.append(result)
                continue

        close_time = window_end if window_end is not None else last_epoch
        results.append(
            {
                "turn_index": idx,
                "open_duration_ms": close_time - window_start,
                "time_to_complete_ms": None,
                "completion": "stuck",
            }
        )

    return results


def _turn_stop_strategy(output_dir: str) -> str | None:
    """Return this run's configured model.turn_stop_strategy, or None if config.json wasn't found."""
    config = _find_run_config(output_dir)
    if config is None:
        return None
    model_cfg = config.get("model") or {}
    return model_cfg.get("turn_stop_strategy", "turn_analyzer")


def compute_vad_turn_sub_metrics(context: MetricContext) -> tuple[dict[str, MetricScore], list[dict[str, Any]]] | None:
    """Compute the VAD-analyzer sub-metrics plus the per-turn diagnostic list.

    Returns None when the run used no local turn-analyzer VAD (vad_turn_metrics_applicable
    is False) or recorded no VAD-analyzer turns at all.
    """
    if not vad_turn_metrics_applicable(context.output_dir):
        return None

    vad_starts, vad_stops = _load_vad_events(context.output_dir)
    if not vad_starts:
        return None

    turn_metrics_events = _load_turn_metrics_events(context.output_dir)
    stop_secs_silence_ms = _load_stop_secs_silence_ms(context.output_dir)
    transcripts = _load_transcripts(context.output_dir)
    per_turn = _classify_turns(vad_starts, vad_stops, turn_metrics_events, stop_secs_silence_ms, transcripts)

    def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
        return MetricScore(name=f"turn_taking.{key}", score=value, normalized_score=value if normalized else None)

    sub: dict[str, MetricScore] = {}

    eot_not_fired = (
        1.0
        if per_turn[-1]["completion"] == "stuck"
        and is_agent_timeout_on_user_turn(
            context.conversation_ended_reason,
            context.audio_timestamps_user_turns,
            context.audio_timestamps_assistant_turns,
        )
        else 0.0
    )
    sub["eot_vad_not_fired_rate"] = _wrap("eot_vad_not_fired_rate", eot_not_fired, True)

    natural_count = sum(1 for t in per_turn if t["completion"] == "natural")
    forced_count = sum(1 for t in per_turn if t["completion"] == "forced")
    strategy = _turn_stop_strategy(context.output_dir)
    if strategy != "krisp_viva_turn" and (natural_count + forced_count) > 0:
        sub["forced_completion_rate"] = _wrap(
            "forced_completion_rate", round(forced_count / (natural_count + forced_count), 4), True
        )

    # Among forced completions specifically: does the final utterance's shape (short /
    # acknowledgement / spelled-out entity) explain why the turn analyzer needed the stop_secs
    # fallback instead of closing naturally? Only emitted when there's at least one forced
    # completion with a matched transcript, so cross-record aggregates reflect real evidence.
    forced_flags = [t["final_turn_flags"] for t in per_turn if t["completion"] == "forced" and "final_turn_flags" in t]
    if forced_flags:
        for key in _FINAL_TURN_FLAG_KEYS:
            rate = sum(1 for flags in forced_flags if flags[key]) / len(forced_flags)
            sub[f"forced_completion_final_turn_{key}_rate"] = _wrap(
                f"forced_completion_final_turn_{key}_rate", round(rate, 4), True
            )

    open_durations = [t["open_duration_ms"] for t in per_turn]
    sub["max_turn_open_duration_ms"] = _wrap("max_turn_open_duration_ms", round(max(open_durations), 3), False)

    natural_times = [
        t["time_to_complete_ms"]
        for t in per_turn
        if t["completion"] == "natural" and t["time_to_complete_ms"] is not None
    ]
    if natural_times:
        sorted_times = sorted(natural_times)
        n = len(sorted_times)

        def _pct(p: float) -> float:
            return sorted_times[min(n - 1, int(p * n))]

        sub["mean_time_to_complete_ms"] = _wrap(
            "mean_time_to_complete_ms", round(statistics.mean(natural_times), 3), False
        )
        sub["p50_time_to_complete_ms"] = _wrap("p50_time_to_complete_ms", round(_pct(0.50), 3), False)
        sub["p90_time_to_complete_ms"] = _wrap("p90_time_to_complete_ms", round(_pct(0.90), 3), False)

    return sub, per_turn
