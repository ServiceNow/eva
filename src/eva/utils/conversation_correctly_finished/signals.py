"""Shared data contract: ``ConvFinishSignals`` (signals extracted from a record) and ``Classification`` (the result)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConvFinishSignals:
    """Everything the classifier needs, extracted once from a record's raw files.

    Defaults describe a *clean* parent-failure baseline (nothing detected → ``unknown_reason``).
    """

    # --- parent gate -------------------------------------------------------
    is_parent_failure: bool = True  # inactivity_timeout AND user was last audio speaker
    is_cascade: bool = True  # CASCADE stack (vs S2S / AUDIO_LLM)
    last_conv_message_role: str | None = "assistant"

    # --- reasoning_only / reasoning_too_long (agent_perf_stats last row) ----
    last_perf_response_empty: bool = False
    last_perf_has_tool_call: bool = False
    last_perf_reasoning: str = ""
    last_perf_reasoning_tokens: int = 0  # hidden-reasoning models report tokens but no text
    last_perf_stop_reason: str = ""  # 'length' ⇒ hit the token cap (reasoning_too_long)
    num_llm_calls: int = 0

    # --- llm_api_error (logs.log litellm patterns) -------------------------
    llm_api_error_terminal: bool = False
    llm_error_type: str = ""
    llm_error_excerpt: str = ""
    num_llm_api_error_lines: int = 0

    # --- tts / stt service errors (pipecat error frames) -------------------
    tts_service_error: bool = False
    stt_service_error: bool = False
    service_error_name: str = ""
    service_error_excerpt: str = ""
    num_service_error_frames: int = 0
    assistant_audio_events: int = 1

    # --- stt no-transcription (logs.log) -----------------------------------
    vad_no_tx_warning_after_last_response: bool = False
    empty_transcript_after_last_response: bool = False
    num_vad_no_tx_warnings: int = 0

    # --- ended_with_user_interruption (logs.log) ---------------------------
    num_interruptions: int = 0
    accepted_turn_before_last_interruption: str | None = None
    response_after_last_interruption: bool = True

    # --- vad_no_turn_detected (sim events vs pipecat VAD) ------------------
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
