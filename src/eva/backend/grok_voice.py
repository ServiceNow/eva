"""Grok Voice ``Backend``: xAI's voice realtime API (OpenAI Realtime-compatible).

xAI's voice realtime API is event-compatible with OpenAI's Realtime API
(https://docs.x.ai/developers/model-capabilities/audio/voice-agent), so this
backend subclasses ``OpenAIRealtimeBackend`` and overrides only what differs --
mirroring how ``eva.assistant.grok_voice_server.GrokVoiceAssistantServer``
subclasses the OpenAI Realtime server:

- endpoint: point the client at ``https://api.x.ai/v1`` (default ``base_url``);
- default voice: xAI's built-in voices (``eve``/``ara``/``rex``/``sal``/``leo``);
- api key: falls back to ``XAI_API_KEY`` (not ``OPENAI_API_KEY``, which would be
  the wrong key for x.ai) when not supplied in config;
- input transcription: xAI fires
  ``conversation.item.input_audio_transcription.completed`` multiple times per
  turn, each carrying the *cumulative* text so far rather than a delta. Only the
  latest cumulative is buffered on the session and emitted as ONE input
  ``TRANSCRIPT`` at the turn boundary (next ``speech_started`` / ``response.done``),
  matching the legacy ``grok_voice_server``'s deferred flush -- so the role logs a
  single user turn instead of one per fragment.

Everything else -- session assembly, audio, tool round-trip, interruption,
usage, the rest of the event normalization -- is inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from eva.backend.base import BackendEvent, BackendEventType
from eva.backend.openai_realtime import OpenAIRealtimeBackend, OpenAIRealtimeSession
from eva.utils.logging import get_logger

logger = get_logger(__name__)

XAI_REALTIME_BASE_URL = "https://api.x.ai/v1"
DEFAULT_VOICE = "eve"


@dataclass
class GrokVoiceSession(OpenAIRealtimeSession):
    """OpenAI Realtime session state plus xAI's buffered cumulative input transcript.

    xAI streams ``input_audio_transcription.completed`` repeatedly, each carrying the
    *cumulative* text so far (it does NOT emit incremental ``.delta`` events like
    OpenAI). Emitting each ``completed`` directly makes the role log one user turn per
    fragment. Instead ``pending_input_transcript`` holds the latest cumulative text,
    which is flushed as a single input ``TRANSCRIPT`` at the turn boundary.
    """

    pending_input_transcript: str = ""


class GrokVoiceBackend(OpenAIRealtimeBackend):
    """xAI Grok voice realtime behind the role-agnostic ``Backend`` contract."""

    _SESSION_CLS: ClassVar[type[OpenAIRealtimeSession]] = GrokVoiceSession
    _API_KEY_ENV: ClassVar[str] = "XAI_API_KEY"

    # Turn boundaries at which the buffered cumulative input transcript is flushed
    # (mirrors the legacy server flushing on _on_speech_started / _on_response_done).
    _FLUSH_ON: ClassVar[tuple[str, ...]] = ("input_audio_buffer.speech_started", "response.done")

    def __init__(self, *, config: dict[str, Any]) -> None:
        # base_url / speaker_id are xAI defaults any explicit config overrides; api_key
        # falls back to XAI_API_KEY in the parent (via _API_KEY_ENV).
        merged = {"base_url": XAI_REALTIME_BASE_URL, "speaker_id": DEFAULT_VOICE, **config}
        super().__init__(config=merged)

    @staticmethod
    def _map_event(session: OpenAIRealtimeSession, event: Any) -> list[BackendEvent]:
        """Normalize one xAI event, buffering its cumulative input transcript.

        ``input_audio_transcription.completed`` is cumulative (xAI re-sends the whole
        text, no ``.delta`` events), so it is buffered rather than emitted; the buffer
        is flushed as one input ``TRANSCRIPT`` just before the parent's events at a turn
        boundary (``_FLUSH_ON``). Everything else defers to ``OpenAIRealtimeBackend``.
        """
        etype = getattr(event, "type", "")
        if etype == "conversation.item.input_audio_transcription.completed":
            text = (getattr(event, "transcript", "") or "").strip()
            if isinstance(session, GrokVoiceSession) and text:
                session.pending_input_transcript = text
            return []

        events: list[BackendEvent] = []
        if etype in GrokVoiceBackend._FLUSH_ON:
            events.extend(GrokVoiceBackend._flush_input_transcript(session))
        events.extend(OpenAIRealtimeBackend._map_event(session, event))
        return events

    @staticmethod
    def _flush_input_transcript(session: OpenAIRealtimeSession) -> list[BackendEvent]:
        """Emit the buffered cumulative input transcript once, then clear it."""
        if not isinstance(session, GrokVoiceSession) or not session.pending_input_transcript:
            return []
        text = session.pending_input_transcript
        session.pending_input_transcript = ""
        return [
            BackendEvent(
                event_type=BackendEventType.TRANSCRIPT,
                transcript=text,
                metadata={"stream": "input", "final": True},
            )
        ]
