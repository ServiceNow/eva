"""Causes: the user spoke but the agent never got the turn.

STT gave no usable transcript (``stt_missing_transcription`` / ``stt_empty_transcription``) or VAD
never fired for the final utterance (``vad_no_turn_detected``).
"""

import re
from typing import Any

from eva.utils.conversation_correctly_finished.signals import Classification, ConvFinishSignals

_RE_VADNOTX = re.compile(r"VAD fired but no previous transcription timestamp found")
_RE_TXSTOP_EMPTY = re.compile(r"User turn stopped - complete transcript: ''")
_RE_STILL = re.compile(r"Assistant still speaking - resetting inactivity")
_VAD_WINDOW_S = 1.5


def extract_vad_onsets(pipecat_events: list[dict]) -> list[float]:
    """VAD onset timestamps (seconds) from pipecat ``user_started_speaking`` events."""
    return [e["timestamp"] / 1000.0 for e in pipecat_events if e.get("type") == "user_started_speaking"]


def detect_final_utterance_vad(
    s: ConvFinishSignals,
    u_starts: list[float],
    u_ends: list[float],
    asst_ends: list[float],
    vad_onsets: list[float],
) -> None:
    """Whether the user's final emitted utterance came after the agent with no VAD onset in-window."""
    if not (u_starts and asst_ends):
        return
    fu_start = max(u_starts)
    fu_end = max([e for e in u_ends if e >= fu_start], default=fu_start + 1.0)
    s.user_final_utterance_after_agent = fu_start >= max(asst_ends) - 0.5
    s.vad_onset_in_final_window = any(fu_start - _VAD_WINDOW_S <= t <= fu_end + _VAD_WINDOW_S for t in vad_onsets)


def extract_log_signals(lines: list[str], resp_idx: list[int], s: ConvFinishSignals) -> None:
    """STT no-transcription: a VAD warning / empty transcript after the last real response.

    Also records ``assistant_still_speaking_before`` (a VAD-cause signal read from the same lines).
    """
    s.assistant_still_speaking_before = any(_RE_STILL.search(ln) for ln in lines)
    last_resp = resp_idx[-1] if resp_idx else -1
    vadnotx_idx = [i for i, ln in enumerate(lines) if _RE_VADNOTX.search(ln)]
    empty_tx_idx = [i for i, ln in enumerate(lines) if _RE_TXSTOP_EMPTY.search(ln)]
    s.num_vad_no_tx_warnings = len(vadnotx_idx)
    s.vad_no_tx_warning_after_last_response = any(i > last_resp for i in vadnotx_idx)
    s.empty_transcript_after_last_response = any(i > last_resp for i in empty_tx_idx)


def detect_stt_no_transcription(s: ConvFinishSignals, base: dict[str, Any]) -> Classification | None:
    """STT gave no usable transcription for a turn VAD detected (empty vs missing transcript)."""
    if not (s.is_cascade and s.last_conv_message_role == "assistant" and s.vad_no_tx_warning_after_last_response):
        return None
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


def detect_vad_no_turn(s: ConvFinishSignals, base: dict[str, Any]) -> Classification | None:
    """VAD never fired for the user's final (emitted) utterance after the agent finished."""
    if not (
        s.user_final_utterance_after_agent
        and not s.vad_onset_in_final_window
        and s.last_conv_message_role == "assistant"
    ):
        return None
    return Classification(
        "vad_no_turn_detected",
        {
            **base,
            "user_said_ground_truth": s.user_final_words,
            "assistant_still_speaking_before": s.assistant_still_speaking_before,
            "vad_onsets_in_window": 0,
        },
    )
