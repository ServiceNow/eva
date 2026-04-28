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

S2S_ASSISTANT_TURNS_DISCLAIMER = ""


def get_user_turns_disclaimer(is_audio_native: bool) -> str:
    """Return the user-turns disclaimer matching the pipeline type."""
    return S2S_USER_TURNS_DISCLAIMER if is_audio_native else CASCADE_USER_TURNS_DISCLAIMER


def get_assistant_turns_disclaimer(is_audio_native: bool) -> str:
    """Return the assistant-turns disclaimer matching the pipeline type.

    Empty string for speech-to-speech, since assistant turns there are not the
    intended-text/TTS-source distinction that cascade has.
    """
    return S2S_ASSISTANT_TURNS_DISCLAIMER if is_audio_native else CASCADE_ASSISTANT_TURNS_DISCLAIMER
