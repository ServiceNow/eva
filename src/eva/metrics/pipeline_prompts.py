"""Pipeline-aware prompt fragments shared across LLM judge metrics.

These disclaimers describe the relationship between the conversation trace and
what the assistant LLM actually saw or produced. The trace builder already uses
``intended`` text for assistant turns and ``transcribed`` text for user turns
(in cascade mode), so the judge is evaluating the LLM on its own input/output —
not on round-trip TTS+STT artifacts. The disclaimers make that explicit to the
judge so transcription errors are not mis-attributed to the LLM.
"""

CASCADE_USER_TURNS_DISCLAIMER = (
    "**About user turns:** User turns are **transcripts** produced by the assistant's speech-to-text (STT) "
    "system. The assistant receives these transcripts as text input — this is the only representation of "
    "user speech available to the assistant. STT transcripts may contain errors (misheard words, garbled "
    "names, dropped syllables), but the assistant cannot know what the user actually said beyond what the "
    "transcript shows. Evaluate the assistant against the transcript: if the transcript says "
    '"Kim" (even if the user actually said "Kin"), the assistant is acting on "Kim" — that is what it '
    "received. Do not penalize the assistant for the transcript's accuracy."
)

S2S_USER_TURNS_DISCLAIMER = (
    "**About user turns:** This is a **speech-to-speech** system — the assistant receives raw audio "
    "directly, not a text transcript. The user turns shown here are the **intended text** (what the user "
    "simulator was instructed to say), not what the assistant heard. The assistant is responsible for its "
    "own audio understanding. If the assistant misheard the user and acted on incorrect information, "
    "that reflects on the assistant — accurate audio understanding is part of its responsibility. The "
    "only mitigation is proper disambiguation: if the assistant was unsure about what it heard, it "
    "should have asked the user to confirm or clarify."
)

CASCADE_ASSISTANT_TURNS_DISCLAIMER = (
    "**About assistant turns:** Assistant turns shown here are the LLM's **intended text** — exactly "
    "what the agent produced before TTS rendering. When a user response in the transcript appears to "
    "dispute, contradict, or react oddly to an assistant turn that itself looks correct, the most likely "
    "cause is an STT error on the user side (the user actually heard something different from what the "
    "transcript shows the assistant said). Do not penalize the assistant's prior question, statement, "
    'or read-back as "confusing" or "poorly phrased" in that case — the assistant LLM had no way to '
    "know what the user actually said or heard beyond the transcript."
)

S2S_ASSISTANT_TURNS_DISCLAIMER = (
    "**About assistant turns:** This is a **speech-to-speech** system — the agent produces audio directly, "
    "with no separate intended-text step. The assistant turns shown here are **STT transcriptions of the "
    "agent's audio**, not text the LLM wrote. Audio articulation fidelity (whether the agent *spoke* an "
    "entity clearly and correctly) is scored separately by the `agent_speech_fidelity` metric on the "
    "actual audio — do not penalize the agent here for what may be TTS-rendering or STT-transcription "
    "artifacts in its turns. Tool call parameters and tool responses shown in the trace are the literal "
    "values the agent sent and received via the API, not audio — if a tool parameter looks wrong, the "
    "agent really sent it that way; if the agent's claim contradicts a tool response, the tool truly "
    "returned the value shown."
)


# Per-dimension, pipeline-specific scoping notes. Empty string for pipelines where
# no carve-out is needed; injected via dedicated placeholders in the judge prompt.

S2S_MISREPRESENTATION_NOTE = (
    "**Speech-to-speech scoping for this dimension.** Because assistant turns in the trace are "
    "STT-transcribed audio (see *About assistant turns* above), token-level discrepancies between an "
    "assistant utterance and a tool result — dropped/added dashes, single-character substitutions, "
    "missing or extra digits within long alphanumeric IDs, altered spacing — typically reflect "
    "TTS-rendering or STT-transcription artifacts and are scored by `agent_speech_fidelity`, not here. "
    "Only flag `misrepresenting_tool_result` when the discrepancy is structural/semantic (wrong field, "
    "wrong order of magnitude, wrong category) or when downstream signals — subsequent tool calls, "
    "follow-up actions, user objections — show the agent was internally operating on a wrong value."
)

S2S_INFORMATION_LOSS_NOTE = (
    "**Speech-to-speech scoping for this dimension.** Because assistant turns in the trace are "
    "STT-transcribed audio (see *About assistant turns* above), variant token-level readings of the "
    "same alphanumeric identifier across nearby assistant turns — dropped/added dashes, single-character "
    "substitutions, missing or extra digits within long IDs, altered spacing or capitalization — typically "
    "reflect TTS-rendering or STT-transcription artifacts on a value the agent is reading consistently in "
    "audio. These are scored by `agent_speech_fidelity`, not here. "
    "Only flag `information_loss` when the discrepancy is structural/semantic (different entity, wrong "
    "field, wrong category — e.g., addressing the user by an entirely different first name, or referencing "
    "a different person/record than the tool returned), or when downstream signals — subsequent tool calls "
    "made with a wrong value, follow-up actions taken on stale data, user objections that the agent then "
    "fails to incorporate — show the agent was internally operating on a wrong value or had genuinely lost "
    "track of the established fact."
)


def get_user_turns_disclaimer(is_audio_native: bool) -> str:
    """Return the user-turns disclaimer matching the pipeline type."""
    return S2S_USER_TURNS_DISCLAIMER if is_audio_native else CASCADE_USER_TURNS_DISCLAIMER


def get_assistant_turns_disclaimer(is_audio_native: bool) -> str:
    """Return the assistant-turns disclaimer matching the pipeline type."""
    return S2S_ASSISTANT_TURNS_DISCLAIMER if is_audio_native else CASCADE_ASSISTANT_TURNS_DISCLAIMER


def get_misrepresentation_pipeline_note(is_audio_native: bool) -> str:
    """Return the pipeline-specific scoping note for the misrepresenting_tool_result dimension."""
    return S2S_MISREPRESENTATION_NOTE if is_audio_native else ""


def get_information_loss_pipeline_note(is_audio_native: bool) -> str:
    """Return the pipeline-specific scoping note for the information_loss dimension."""
    return S2S_INFORMATION_LOSS_NOTE if is_audio_native else ""
