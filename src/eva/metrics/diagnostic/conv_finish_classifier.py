"""Shared classifier for `conversation_correctly_finished` failures.

Splits the single "agent went silent after the user spoke" failure bucket into one primary
cause. Two layers:

- ``ConvFinishSignals`` + ``classify_conv_finish_failure`` — **pure logic**: given the
  extracted signals, apply a fixed priority order and return exactly one category (or ``None``
  when the record isn't a parent failure at all). Unit-tested without any file I/O.
- ``extract_conv_finish_signals`` — the I/O layer that reads the record's raw files and builds
  a ``ConvFinishSignals``. (Kept separate so the logic stays trivially testable.)

Design rationale, per-category detection, and validation evidence live in
``docs/metrics/conv_finish_submetrics.md``.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.models.results import MetricScore

# Priority order: infra failures first (they invalidate the run and supersede apparent
# behaviour), then model/pipeline-specific causes, then the residual catch-all.
CATEGORY_PRIORITY = [
    "tts_api_error",
    "stt_api_error",
    "llm_api_error",
    "answer_lost_in_reasoning",
    "stt_empty_transcription",
    "stt_missing_transcription",
    "ended_with_user_interruption",
    "vad_no_turn_detected",
    "unknown_reason",
]

# Categories that indicate an environmental/infra failure — the agent was never actually
# exercised, so the record should be treated as an invalid run, not an agent regression.
INFRA_CATEGORIES = frozenset({"llm_api_error", "tts_api_error", "stt_api_error"})


@dataclass
class ConvFinishSignals:
    """Everything the classifier needs, extracted once from a record's raw files.

    Defaults describe a *clean* parent-failure baseline (nothing detected → ``unknown_reason``).
    """

    # --- parent gate -------------------------------------------------------
    is_parent_failure: bool = True  # inactivity_timeout AND user was last audio speaker
    is_cascade: bool = True  # CASCADE stack (vs S2S / AUDIO_LLM)
    last_conv_message_role: str | None = "assistant"

    # --- #1 answer_lost_in_reasoning (agent_perf_stats last row) -----------
    last_perf_response_empty: bool = False
    last_perf_has_tool_call: bool = False
    last_perf_reasoning: str = ""
    num_llm_calls: int = 0

    # --- #5 llm_api_error (logs.log litellm patterns) ----------------------
    llm_api_error_terminal: bool = False
    llm_error_type: str = ""
    llm_error_excerpt: str = ""
    num_llm_api_error_lines: int = 0

    # --- #8 / #9 service errors (pipecat error frames) ---------------------
    tts_service_error: bool = False
    stt_service_error: bool = False
    service_error_name: str = ""
    service_error_excerpt: str = ""
    num_service_error_frames: int = 0
    assistant_audio_events: int = 1

    # --- #2 / #3 stt no-transcription (logs.log) ---------------------------
    vad_no_tx_warning_after_last_response: bool = False
    empty_transcript_after_last_response: bool = False
    num_vad_no_tx_warnings: int = 0

    # --- #4 ended_with_user_interruption (logs.log) ------------------------
    num_interruptions: int = 0
    accepted_turn_before_last_interruption: str | None = None
    response_after_last_interruption: bool = True
    interruption_delay_after_turn_ms: int | None = None

    # --- #7 vad_no_turn_detected (sim events vs pipecat VAD) ----------------
    user_final_utterance_after_agent: bool = False
    vad_onset_in_final_window: bool = True
    user_final_words: str | None = None
    assistant_still_speaking_before: bool = False

    # extraction problems (missing files); still classifiable, noted in details
    notes: list[str] = field(default_factory=list)


@dataclass
class Classification:
    category: str | None  # None when the record is not a parent failure
    details: dict[str, Any]


def classify_conv_finish_failure(s: ConvFinishSignals) -> Classification:
    """Return the single primary cause for a conv_finish failure (or ``None`` if not one)."""
    if not s.is_parent_failure:
        return Classification(None, {})

    base: dict[str, Any] = {}
    if s.notes:
        base["notes"] = list(s.notes)

    # 1-3. Infra / service errors (highest priority; mark invalid run).
    if s.tts_service_error:
        return Classification("tts_api_error", {**base, **_infra_details("tts", s)})
    if s.stt_service_error:
        return Classification("stt_api_error", {**base, **_infra_details("stt", s)})
    if s.llm_api_error_terminal:
        return Classification(
            "llm_api_error",
            {
                **base,
                "invalid_run": True,
                "component": "llm",
                "error_type": s.llm_error_type,
                "error_excerpt": s.llm_error_excerpt,
                "num_api_error_lines": s.num_llm_api_error_lines,
                "responses_after_last_error": 0,
            },
        )

    # 4. Answer trapped in the reasoning field (reasoning stacks).
    if s.last_perf_response_empty and not s.last_perf_has_tool_call and s.last_perf_reasoning.strip():
        return Classification(
            "answer_lost_in_reasoning",
            {
                **base,
                "final_response_empty": True,
                "final_row_had_tool_call": False,
                "recovered_answer": s.last_perf_reasoning[:300],
                "reasoning_chars": len(s.last_perf_reasoning),
                "num_llm_calls": s.num_llm_calls,
            },
        )

    # 5-6. STT gave no usable transcription for a turn VAD detected.
    if s.is_cascade and s.last_conv_message_role == "assistant" and s.vad_no_tx_warning_after_last_response:
        category = "stt_empty_transcription" if s.empty_transcript_after_last_response else "stt_missing_transcription"
        return Classification(
            category,
            {
                **base,
                "vad_fired": True,
                "final_user_turn_in_llm_context": False,
                "user_turn_stopped_emitted": s.empty_transcript_after_last_response,
                "user_said_ground_truth": s.user_final_words,
                "num_vad_no_transcription_warnings": s.num_vad_no_tx_warnings,
            },
        )

    # 7. Response cancelled by an interruption and never resumed.
    if s.num_interruptions > 0 and s.accepted_turn_before_last_interruption and not s.response_after_last_interruption:
        return Classification(
            "ended_with_user_interruption",
            {
                **base,
                "accepted_user_turn": s.accepted_turn_before_last_interruption,
                "interruption_delay_after_turn_ms": s.interruption_delay_after_turn_ms,
                "num_interruptions": s.num_interruptions,
                "responses_after_last_interruption": 0,
            },
        )

    # 8. VAD never fired for the user's final (emitted) utterance.
    if (
        s.user_final_utterance_after_agent
        and not s.vad_onset_in_final_window
        and s.last_conv_message_role == "assistant"
    ):
        return Classification(
            "vad_no_turn_detected",
            {
                **base,
                "user_said_ground_truth": s.user_final_words,
                "assistant_still_speaking_before": s.assistant_still_speaking_before,
                "vad_onsets_in_window": 0,
            },
        )

    # 9. Residual — unexplained failure; candidate for a new sub-metric.
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


# --- extraction (I/O layer) -------------------------------------------------

csv.field_size_limit(10**7)  # agent_perf_stats rows embed full prompts

_RE_RESP = re.compile(r"complete response: '(.*)'$")
_RE_VADNOTX = re.compile(r"VAD fired but no previous transcription timestamp found")
_RE_TXSTOP_EMPTY = re.compile(r"User turn stopped - complete transcript: ''")
_RE_PROC = re.compile(r"Processing complete user turn: (.+)$")
_RE_INTR = re.compile(r"Interruption received - cancelling ongoing")
_RE_STILL = re.compile(r"Assistant still speaking - resetting inactivity")
# Fatal LLM API-error signatures — allowlist, not a broad `litellm.*Error` match.
_RE_LLM_ERR = re.compile(
    r"MidStreamFallbackError|APIConnectionError|Retryable streaming error|RateLimitError|InternalServerError"
)
_RE_SERVICE = re.compile(r"([A-Za-z0-9]+(?:TTS|STT)Service)")
_VAD_WINDOW_S = 1.5


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


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

    # audit_log → last conversation message role
    audit = out / "audit_log.json"
    if audit.exists():
        try:
            cm = json.loads(audit.read_text()).get("conversation_messages", [])
            s.last_conv_message_role = cm[-1]["role"] if cm else None
        except (json.JSONDecodeError, KeyError):
            s.notes.append("audit_log unreadable")

    # agent_perf_stats.csv → last row (answer_lost signals)
    perf = out / "agent_perf_stats.csv"
    if perf.exists():
        try:
            rows = list(csv.DictReader(perf.open(newline="")))
            s.num_llm_calls = len(rows)
            if rows:
                last = rows[-1]
                s.last_perf_response_empty = not (last.get("response") or "").strip()
                s.last_perf_has_tool_call = bool((last.get("tool_calls") or "").strip())
                s.last_perf_reasoning = (last.get("reasoning") or "").strip()
        except (OSError, csv.Error):
            s.notes.append("agent_perf_stats unreadable")

    # user_simulator_events → ground-truth audio timeline + final words + assistant audio count
    ev = _load_jsonl(out / "user_simulator_events.jsonl")
    user_speech = [e for e in ev if e.get("type") == "user_speech"]
    s.user_final_words = user_speech[-1]["data"]["text"] if user_speech else None
    asst_ends = [
        e["audio_timestamp"] for e in ev if e.get("event_type") == "audio_end" and e.get("user") == "assistant"
    ]
    s.assistant_audio_events = sum(
        1 for e in ev if e.get("event_type") == "audio_start" and e.get("user") == "assistant"
    )
    u_starts = [
        e["audio_timestamp"] for e in ev if e.get("event_type") == "audio_start" and e.get("user") == "simulated_user"
    ]
    u_ends = [
        e["audio_timestamp"] for e in ev if e.get("event_type") == "audio_end" and e.get("user") == "simulated_user"
    ]

    # pipecat_logs → service-error frames + VAD onsets
    pl = _load_jsonl(out / "pipecat_logs.jsonl")
    for e in pl:
        if e.get("type") == "error":
            frame = str(e.get("data", {}).get("frame", ""))
            m = _RE_SERVICE.search(frame)
            if m:
                s.num_service_error_frames += 1
                svc = m.group(1)
                if "TTS" in svc:
                    s.tts_service_error = True
                    s.service_error_name = svc
                elif "STT" in svc:
                    s.stt_service_error = True
                    s.service_error_name = svc
                s.service_error_excerpt = frame[:160]
    vad_onsets = [e["timestamp"] / 1000.0 for e in pl if e.get("type") == "user_started_speaking"]

    # #7: user's final emitted utterance after the agent, with no VAD onset in its window
    if u_starts and asst_ends:
        fu_start = max(u_starts)
        fu_end = max([e for e in u_ends if e >= fu_start], default=fu_start + 1.0)
        s.user_final_utterance_after_agent = fu_start >= max(asst_ends) - 0.5
        s.vad_onset_in_final_window = any(fu_start - _VAD_WINDOW_S <= t <= fu_end + _VAD_WINDOW_S for t in vad_onsets)

    # logs.log → response/interruption/STT/LLM-error line analysis
    log_path = out / "logs.log"
    if log_path.exists():
        _extract_log_signals(log_path.read_text().splitlines(), s)
    elif not (s.tts_service_error or s.stt_service_error or perf.exists()):
        s.notes.append("no logs.log")

    return s


def _msg(line: str) -> str:
    """The message portion of a log line ('… | msg'), or the whole line."""
    return line.rsplit(" | ", 1)[-1]


def _extract_log_signals(lines: list[str], s: ConvFinishSignals) -> None:
    resp_idx = [i for i, ln in enumerate(lines) if (m := _RE_RESP.search(_msg(ln))) and m.group(1).strip()]
    last_resp = resp_idx[-1] if resp_idx else -1
    intr_idx = [i for i, ln in enumerate(lines) if _RE_INTR.search(ln)]
    proc = [(i, m.group(1).strip()) for i, ln in enumerate(lines) if (m := _RE_PROC.search(_msg(ln)))]
    err_idx = [i for i, ln in enumerate(lines) if _RE_LLM_ERR.search(ln)]
    vadnotx_idx = [i for i, ln in enumerate(lines) if _RE_VADNOTX.search(ln)]
    empty_tx_idx = [i for i, ln in enumerate(lines) if _RE_TXSTOP_EMPTY.search(ln)]
    s.assistant_still_speaking_before = any(_RE_STILL.search(ln) for ln in lines)

    # STT no-transcription: warning/empty-transcript after the last real assistant response
    s.num_vad_no_tx_warnings = len(vadnotx_idx)
    s.vad_no_tx_warning_after_last_response = any(i > last_resp for i in vadnotx_idx)
    s.empty_transcript_after_last_response = any(i > last_resp for i in empty_tx_idx)

    # LLM API error: fatal error with no non-empty response after the LAST error line
    if err_idx:
        last_err = err_idx[-1]
        s.num_llm_api_error_lines = len(err_idx)
        s.llm_api_error_terminal = not any(i > last_err for i in resp_idx)
        if s.llm_api_error_terminal:
            excerpt = _RE_LLM_ERR.search(lines[last_err])
            s.llm_error_type = excerpt.group(0) if excerpt else ""
            s.llm_error_excerpt = _msg(lines[last_err])[:160]

    # Interruption: cancelled a real accepted turn, no response after the last interruption
    if intr_idx:
        last_intr = intr_idx[-1]
        s.num_interruptions = len(intr_idx)
        proc_before = [t for (i, t) in proc if i < last_intr]
        s.accepted_turn_before_last_interruption = proc_before[-1] if proc_before else None
        s.response_after_last_interruption = any(i > last_intr for i in resp_idx)


# Final-turn input-characteristic heuristics — kept in sync with the analysis app
# (EVABench `apps/analysis.py`: `_strip_annotations`, `_CONFIRM_START_RE`, `_spelled_out_signals`).
# Prosody/annotation tags ([slow], [neutral], [user interrupts]) inflate word counts and hide the
# acknowledgement lead, so strip them first.
_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
# Acknowledgement lead — mirrors EVABench `_CONFIRM_START_RE` (includes negations no/nope, tolerant
# of leading [tags]); scope is "starts with a confirmation/acknowledgement".
_ACKNOWLEDGEMENT_LEAD = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)*\b(?:yes|no|nope|yep|yeah|sure|okay|ok|alright|correct|right)\b",
    re.IGNORECASE,
)
# Spelled-entity signals (ported from EVABench `_spelled_out_signals`).
# Single-letter / single-digit runs require ≥3 chars — 2 chars ("I T" = the word IT, "G A") isn't
# spelling.
_SEPS = r"[\s.,;:\-…]+"
_SINGLE_LETTER_RUN_RE = re.compile(rf"(?:\b[a-zA-Z]\b{_SEPS}){{2,}}\b[a-zA-Z]\b")
_SINGLE_DIGIT_RUN_RE = re.compile(rf"(?:\b\d\b{_SEPS}){{2,}}\b\d\b")
_AS_IN_RE = re.compile(r"\b[a-zA-Z]\s+as\s+in\s+\w+", re.IGNORECASE)
_CAPS_ALNUM_CODE_RE = re.compile(r"\b(?=[A-Z0-9]{3,})(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")
_CAPS_ACRONYM_RE = re.compile(r"\b[A-Z]{3,}\b")
_DIGIT_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)"
_SPOKEN_DIGIT_RUN_RE = re.compile(rf"(?:\b{_DIGIT_WORD}\b[^a-zA-Z]+){{2,}}\b{_DIGIT_WORD}\b", re.IGNORECASE)
_NATO_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "x-ray",
    "xray",
    "yankee",
    "zulu",
)
_NATO_WORD_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _NATO_WORDS) + r")\b", re.IGNORECASE)
_NATO_SET = set(_NATO_WORDS)
_DIGIT_WORD_SET = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
# Length caps (annotation-stripped word counts). A genuine acknowledgement is short; a long sentence
# that merely starts "yeah …" isn't one. A spelled entity must *dominate* the turn — so we cap the
# surrounding non-spelled (prose) words rather than total words (real codes are long).
_ACKNOWLEDGEMENT_MAX_WORDS = 6
_SPELL_MAX_EXTRA_WORDS = 10


def _strip_annotations(text: str) -> str:
    return _ANNOTATION_RE.sub(" ", text or "")


def _non_spell_word_count(text: str) -> int:
    """Words that are NOT part of a spell-out (single letters, digit-words, 'dash', NATO, caps codes)."""
    n = 0
    for raw in text.split():
        tok = raw.strip(".,!?;:'\"()[]…-")
        if not tok:
            continue
        low = tok.lower()
        is_spell_tok = (
            (len(tok) == 1 and tok.isalpha())
            or low in _DIGIT_WORD_SET
            or low == "dash"
            or low in _NATO_SET
            or (tok.isupper() and len(tok) >= 3)  # caps acronym (VPN, HIPAA)
            or (len(tok) >= 3 and tok.isalnum() and any(c.isupper() for c in tok) and any(c.isdigit() for c in tok))
        )
        if not is_spell_tok:
            n += 1
    return n


def _looks_spelled(text: str) -> bool:
    """Any spelled-out signal — single-letter/digit runs, NATO, 'X as in Y', caps codes/acronyms."""
    return bool(
        _SINGLE_LETTER_RUN_RE.search(text)
        or _SINGLE_DIGIT_RUN_RE.search(text)
        or _AS_IN_RE.search(text)
        or _NATO_WORD_RE.search(text)
        or _CAPS_ALNUM_CODE_RE.search(text)
        or _CAPS_ACRONYM_RE.search(text)
        or _SPOKEN_DIGIT_RUN_RE.search(text)
    )


def final_turn_input_flags(text: str | None) -> dict[str, bool]:
    """Orthogonal input-characteristic flags for the user's final intended utterance.

    These are NOT pipeline causes — they describe the *input* and cut across the cause partition
    (a failure can be both `vad_no_turn_detected` and have an acknowledgement final turn). Used to
    test whether short / acknowledgement / spelled-out final turns correlate with failures.
    Heuristics mirror the analysis app (EVABench `apps/analysis.py`) so the two tools agree.

    - ``short``: 1-2 words (< 3), counted **after stripping `[annotation]` tags**.
    - ``acknowledgement``: leads with a confirmation/ack ("Yes, that is correct.", "Ok thanks",
      "[neutral] Okay.", "No.") AND is at most ``_ACKNOWLEDGEMENT_MAX_WORDS`` words — a long sentence
      that merely starts "yeah …" is not counted.
    - ``spelled_entity``: letter/digit spell-out of an ID/code/name ("E M P eight nine …", NATO,
      "V as in Victor", caps codes like EMP358) that **dominates** the turn (≤ ``_SPELL_MAX_EXTRA_WORDS``
      non-spelled words) — a long sentence with a small spelled fragment is not counted. Single-char
      runs need ≥3 chars, so 2-letter words like "IT" ("I T") don't count.
    """
    if not text or not text.strip():
        return {"short": False, "acknowledgement": False, "spelled_entity": False}
    stripped = _strip_annotations(text).strip()
    word_count = len(stripped.split())
    return {
        "short": 0 < word_count < 3,
        "acknowledgement": word_count <= _ACKNOWLEDGEMENT_MAX_WORDS and bool(_ACKNOWLEDGEMENT_LEAD.match(stripped)),
        "spelled_entity": _looks_spelled(stripped) and _non_spell_word_count(stripped) <= _SPELL_MAX_EXTRA_WORDS,
    }


def build_final_turn_flag_sub_metrics(
    user_final_text: str | None,
    parent_name: str = "conversation_correctly_finished",
) -> dict[str, "MetricScore"]:
    """Binary flags (``1.0`` = present) for final-turn input characteristics — orthogonal to causes."""
    flags = final_turn_input_flags(user_final_text)
    subs: dict[str, MetricScore] = {}
    for key in ("short", "acknowledgement", "spelled_entity"):
        occurred = flags[key]
        score = 1.0 if occurred else 0.0
        sub_key = f"final_turn_{key}_rate"
        subs[sub_key] = MetricScore(
            name=f"{parent_name}.{sub_key}",
            score=score,
            normalized_score=score,
            details={"final_user_text": user_final_text} if occurred else {},
        )
    return subs


def build_conv_finish_sub_metrics(
    classification: Classification,
    parent_name: str = "conversation_correctly_finished",
) -> dict[str, "MetricScore"]:
    """Per-cause binary-flag sub-metrics (framework convention: ``1.0`` = this cause occurred).

    One flag per category in ``CATEGORY_PRIORITY``, keyed ``<cause>_rate`` (the ``_rate`` suffix
    signals lower-is-better to ``direction_for_sub_metric`` and makes the cross-record mean read
    as "fraction of failures with this cause"). The fired cause carries the evidence details.
    """
    subs: dict[str, MetricScore] = {}
    for cause in CATEGORY_PRIORITY:
        occurred = cause == classification.category
        score = 1.0 if occurred else 0.0
        sub_key = f"{cause}_rate"
        subs[sub_key] = MetricScore(
            name=f"{parent_name}.{sub_key}",
            score=score,
            normalized_score=score,
            details=classification.details if occurred else {},
        )
    return subs


def _infra_details(component: str, s: ConvFinishSignals) -> dict[str, Any]:
    return {
        "invalid_run": True,
        "component": component,
        "service": s.service_error_name,
        "error_excerpt": s.service_error_excerpt,
        "num_error_frames": s.num_service_error_frames,
        "assistant_audio_events": s.assistant_audio_events,
    }
