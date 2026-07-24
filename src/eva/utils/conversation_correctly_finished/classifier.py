"""Core pipeline: extract signals from a record, classify the primary cause, build sub-metric flags.

Three stages, top to bottom:

1. **Extraction** (I/O) — read a record's raw files into a ``ConvFinishSignals``. The orchestrator
   reads each file once, sets the shared fields, and hands pre-parsed data to each ``causes/*``
   extractor (which owns that cause's signatures).
2. **Classification** (pure logic, no I/O) — order the per-cause ``detect_*`` functions by priority
   and return the single primary cause (or ``None`` when the record isn't a parent failure).
3. **Sub-metric builders** — turn a ``Classification`` / final-turn text into ``MetricScore`` flags.
"""

import re
from pathlib import Path
from typing import Any

from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.models.results import MetricScore
from eva.utils.conversation_correctly_finished.causes import api_errors, interruption, reasoning, transcript_vad
from eva.utils.conversation_correctly_finished.final_turn import final_turn_input_flags
from eva.utils.conversation_correctly_finished.signals import Classification, ConvFinishSignals
from eva.utils.json_utils import load_jsonl
from eva.utils.log_processing import load_audit_log, parse_log_message

# Shared "the agent produced a real spoken response" signal — several causes ask "was there a
# response after event X?", so its line index is computed once and passed to each extractor.
_RE_RESP = re.compile(r"complete response: '(.*)'$")


# --- extraction (I/O) -------------------------------------------------------


def _extract_audit_log(audit: Path, s: ConvFinishSignals) -> None:
    """Last conversation message role, from audit_log.json."""
    if not audit.exists():
        return
    data = load_audit_log(audit)
    if data is None:
        s.notes.append("audit_log unreadable")
        return
    try:
        cm = data.get("conversation_messages", [])
        s.last_conv_message_role = cm[-1]["role"] if cm else None
    except KeyError:
        s.notes.append("audit_log unreadable")


def _extract_sim_audio(out: Path, s: ConvFinishSignals) -> tuple[list[float], list[float], list[float]]:
    """User's final words + assistant-audio count; returns (user_starts, user_ends, assistant_ends)."""
    ev = load_jsonl(out / "user_simulator_events.jsonl")
    user_speech = [e for e in ev if e.get("type") == "user_speech"]
    s.user_final_words = user_speech[-1]["data"]["text"] if user_speech else None
    s.assistant_audio_events = sum(
        1 for e in ev if e.get("event_type") == "audio_start" and e.get("user") == "assistant"
    )
    asst_ends = [
        e["audio_timestamp"] for e in ev if e.get("event_type") == "audio_end" and e.get("user") == "assistant"
    ]
    u_starts = [
        e["audio_timestamp"] for e in ev if e.get("event_type") == "audio_start" and e.get("user") == "simulated_user"
    ]
    u_ends = [
        e["audio_timestamp"] for e in ev if e.get("event_type") == "audio_end" and e.get("user") == "simulated_user"
    ]
    return u_starts, u_ends, asst_ends


def _extract_log_signals(lines: list[str], s: ConvFinishSignals) -> None:
    """Parse logs.log once into the shared ``resp_idx``, then fan out to per-cause extractors."""
    resp_idx = [i for i, ln in enumerate(lines) if (m := _RE_RESP.search(parse_log_message(ln))) and m.group(1).strip()]
    transcript_vad.extract_log_signals(lines, resp_idx, s)
    api_errors.extract_log_signals(lines, resp_idx, s)
    interruption.extract_log_signals(lines, resp_idx, s)


def extract_conv_finish_signals(context) -> ConvFinishSignals:  # noqa: ANN001 (MetricContext)
    """Read a record's raw files and build the signals the classifier needs."""
    s = ConvFinishSignals()
    s.is_parent_failure = is_agent_timeout_on_user_turn(
        context.conversation_ended_reason,
        context.audio_timestamps_user_turns,
        context.audio_timestamps_assistant_turns,
    )
    s.is_cascade = not context.is_audio_native
    out = Path(context.output_dir) if context.output_dir else None
    if out is None:
        s.notes.append("no output_dir")
        return s

    _extract_audit_log(out / "audit_log.json", s)
    perf = out / "agent_perf_stats.csv"
    reasoning.extract_perf_stats(perf, s)

    u_starts, u_ends, asst_ends = _extract_sim_audio(out, s)
    pipecat_events = load_jsonl(out / "pipecat_logs.jsonl")
    api_errors.extract_service_errors(pipecat_events, s)
    vad_onsets = transcript_vad.extract_vad_onsets(pipecat_events)
    transcript_vad.detect_final_utterance_vad(s, u_starts, u_ends, asst_ends, vad_onsets)

    log_path = out / "logs.log"
    if log_path.exists():
        _extract_log_signals(log_path.read_text().splitlines(), s)
    elif not (s.tts_service_error or s.stt_service_error or perf.exists()):
        s.notes.append("no logs.log")

    return s


# --- classification (pure logic, no I/O) ------------------------------------


def _detect_unknown(s: ConvFinishSignals, base: dict[str, Any]) -> Classification:
    """Residual — no known cause matched; a candidate for a new sub-metric."""
    return Classification(
        "unknown_reason",
        {
            **base,
            "last_conversation_message_role": s.last_conv_message_role,
            "num_interruptions": s.num_interruptions,
            "final_user_turn_transcript": s.user_final_words,
            "note": "no known sub-metric matched — candidate for a new sub-metric",
        },
    )


# The detector partition, in priority order — the first detector to return a Classification wins.
# Each detector is paired with the category names it can emit (also in priority order). Pairing
# them here makes this the single source of truth: CATEGORY_PRIORITY is derived from it, so the
# documented priority can never drift from the order detectors actually run in.
_DETECTORS = (
    (api_errors.detect_service_error, ("tts_api_error", "stt_api_error")),
    (api_errors.detect_llm_api_error, ("llm_api_error",)),
    (reasoning.detect_reasoning, ("reasoning_too_long", "reasoning_only")),
    (transcript_vad.detect_stt_no_transcription, ("stt_empty_transcription", "stt_missing_transcription")),
    (interruption.detect_interruption, ("ended_with_user_interruption",)),
    (transcript_vad.detect_vad_no_turn, ("vad_no_turn_detected",)),
)

# Every category a failure can be assigned, in priority order: derived from _DETECTORS (infra
# causes first — they invalidate the run and supersede apparent behavior) plus the residual
# catch-all, which has no detector and fires iff nothing else did.
CATEGORY_PRIORITY = [category for _, categories in _DETECTORS for category in categories] + ["unknown_reason"]


def classify_conv_finish_failure(s: ConvFinishSignals) -> Classification:
    """Return the single primary cause for a conv_finish failure (or ``None`` if not one)."""
    if not s.is_parent_failure:
        return Classification(None, {})
    base: dict[str, Any] = {"notes": list(s.notes)} if s.notes else {}
    for detect, _categories in _DETECTORS:
        result = detect(s, base)
        if result is not None:
            return result
    return _detect_unknown(s, base)


# --- sub-metric builders ----------------------------------------------------


def _flag_sub_metric(parent_name: str, sub_key: str, occurred: bool, details: dict[str, Any]) -> MetricScore:
    """A binary-flag sub-metric (framework convention: ``1.0`` = the flag fired)."""
    score = 1.0 if occurred else 0.0
    return MetricScore(
        name=f"{parent_name}.{sub_key}",
        score=score,
        normalized_score=score,
        details=details if occurred else {},
    )


def build_final_turn_flag_sub_metrics(
    user_final_text: str | None,
    parent_name: str = "conversation_correctly_finished",
) -> dict[str, MetricScore]:
    """Binary flags (``1.0`` = present) for final-turn input characteristics — orthogonal to causes."""
    flags = final_turn_input_flags(user_final_text)
    details = {"final_user_text": user_final_text}
    return {
        f"final_turn_{key}_rate": _flag_sub_metric(parent_name, f"final_turn_{key}_rate", flags[key], details)
        for key in ("short", "acknowledgement", "spelled_entity")
    }


def build_conv_finish_sub_metrics(
    classification: Classification,
    parent_name: str = "conversation_correctly_finished",
) -> dict[str, MetricScore]:
    """Per-cause binary-flag sub-metrics (framework convention: ``1.0`` = this cause occurred).

    One flag per category in ``CATEGORY_PRIORITY``, keyed ``<cause>_rate`` (the ``_rate`` suffix
    signals lower-is-better to ``direction_for_sub_metric``). Emitted on every record (all 0.0
    when ``classification.category`` is None, i.e. a clean finish), so the cross-record mean reads
    as "fraction of all conversations with this cause" — an all-records denominator matching
    faithfulness. The fired cause carries the evidence details.
    """
    return {
        f"{cause}_rate": _flag_sub_metric(
            parent_name, f"{cause}_rate", cause == classification.category, classification.details
        )
        for cause in CATEGORY_PRIORITY
    }
