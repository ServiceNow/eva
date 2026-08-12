"""Capability flags describing what a ``Backend`` implementation can do.

These flags let a role / turn-taking layer branch on backend shape without every
``Backend`` implementation exposing the same granular seams. ``supports_manual_response``
is consumed by ``UserRole`` to choose between caller-driven manual turn-taking and a
native-VAD path; the others are still informational, reserved for the later
mediator/interruption work.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Static, provider-declared capability flags for a ``Backend``.

    Each concrete ``Backend`` subclass sets these once (typically as a class
    attribute or constructed in ``__init__``) to describe its streaming shape.

    Attributes:
        emits_continuous_audio: True if the backend produces a continuous
            audio stream once speaking starts (native speech-to-speech models
            such as OpenAI Realtime or Gemini Live, and end-to-end providers
            such as ElevenLabs Agents). False for cascade backends
            (STT -> LLM -> TTS) that emit discrete, chunked utterances with
            gaps between TTS renders. This is the flag a future mediator uses
            to decide whether "audio is still arriving" is a meaningful
            signal on its own, or whether it must also track discrete
            utterance boundaries.
        supports_streaming_interruption: True if the underlying provider API
            supports being told "stop talking now" mid-utterance and reacting
            immediately (e.g. OpenAI/Gemini Realtime `response.cancel`-style
            semantics). False if interruption can only be approximated by the
            caller (e.g. stop forwarding audio, drop the rest of a queued TTS
            buffer) rather than being a first-class provider feature. Declared
            for later interruption-policy work; not consumed in this step.
        owns_playout_clock: True if the backend itself is responsible for
            audio playout pacing (e.g. a cascade backend streaming TTS chunks
            at wall-clock rate), which is required for the barge-in work
            planned for a later phase. False if playout pacing is delegated
            to the caller/mediator, or if the backend has no continuous
            playout concept at all (e.g. a fully end-to-end provider that
            hands back a finished audio blob).
        supports_manual_response: True if the provider honors caller-driven manual turn-taking -- i.e. it will
            hold responses when told (``create_response: false``) and only respond when the
            caller explicitly asks (``response.create``). OpenAI Realtime does. Some providers
            (e.g. Grok Voice) ignore this and always auto-respond via their own VAD; they set
            this False so a ``UserRole`` drives them via the native-VAD path (no manual trigger)
            instead of the manual response-gating scheme. Defaults True (manual is preferred
            where available; the alternate path is only for providers that can't do it).
    """

    emits_continuous_audio: bool
    supports_streaming_interruption: bool
    owns_playout_clock: bool
    supports_manual_response: bool = True
