"""Causes: infra API errors — TTS/STT service frames and fatal LLM API errors.

They invalidate the run, so the classifier ranks them ahead of any behavioral cause.
"""

import re
from typing import Any

from eva.utils.conversation_correctly_finished.signals import Classification, ConvFinishSignals
from eva.utils.log_processing import parse_log_message

# Fatal LLM API-error signatures — allowlist, not a broad `litellm.*Error` match.
_RE_LLM_ERR = re.compile(
    r"MidStreamFallbackError|APIConnectionError|Retryable streaming error|RateLimitError|InternalServerError"
)
_RE_SERVICE = re.compile(r"([A-Za-z0-9]+(?:TTS|STT)Service)")


def extract_service_errors(pipecat_events: list[dict], s: ConvFinishSignals) -> None:
    """Set tts/stt service-error flags from pipecat error frames."""
    for e in pipecat_events:
        if e.get("type") != "error":
            continue
        frame = str(e.get("data", {}).get("frame", ""))
        m = _RE_SERVICE.search(frame)
        if not m:
            continue
        s.num_service_error_frames += 1
        svc = m.group(1)
        if "TTS" in svc:
            s.tts_service_error = True
            s.service_error_name = svc
        elif "STT" in svc:
            s.stt_service_error = True
            s.service_error_name = svc
        s.service_error_excerpt = frame[:160]


def extract_log_signals(lines: list[str], resp_idx: list[int], s: ConvFinishSignals) -> None:
    """LLM API error: a fatal error with no non-empty response after the last error line."""
    err_idx = [i for i, ln in enumerate(lines) if _RE_LLM_ERR.search(ln)]
    if not err_idx:
        return
    last_err = err_idx[-1]
    s.num_llm_api_error_lines = len(err_idx)
    s.llm_api_error_terminal = not any(i > last_err for i in resp_idx)
    if s.llm_api_error_terminal:
        m = _RE_LLM_ERR.search(lines[last_err])
        s.llm_error_type = m.group(0) if m else ""
        s.llm_error_excerpt = parse_log_message(lines[last_err])[:160]


def _infra_details(component: str, s: ConvFinishSignals) -> dict[str, Any]:
    """Shared details block for the tts/stt infra-error classifications."""
    return {
        "invalid_run": True,
        "component": component,
        "service": s.service_error_name,
        "error_excerpt": s.service_error_excerpt,
        "num_error_frames": s.num_service_error_frames,
        "assistant_audio_events": s.assistant_audio_events,
    }


def detect_service_error(s: ConvFinishSignals, base: dict[str, Any]) -> Classification | None:
    """Infra: a TTS/STT service raised an error frame — the agent was never exercised."""
    if s.tts_service_error:
        return Classification("tts_api_error", {**base, **_infra_details("tts", s)})
    if s.stt_service_error:
        return Classification("stt_api_error", {**base, **_infra_details("stt", s)})
    return None


def detect_llm_api_error(s: ConvFinishSignals, base: dict[str, Any]) -> Classification | None:
    """Infra: a fatal LLM API error with no successful response after it (invalid run)."""
    if not s.llm_api_error_terminal:
        return None
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
