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
  turn with progressively longer text, so instead of surfacing each as a final
  ``TRANSCRIPT`` (which the OpenAI backend does), buffer it and emit one final
  ``TRANSCRIPT`` when the turn settles (next speech start / ``response.done``).

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
    """OpenAI Realtime session state plus xAI's buffered input transcript.

    ``pending_input_transcript`` accumulates the latest (progressively longer)
    ``input_audio_transcription.completed`` text for the current turn; it is
    flushed as a single final ``TRANSCRIPT`` when the turn settles.
    """

    pending_input_transcript: str = ""


class GrokVoiceBackend(OpenAIRealtimeBackend):
    """xAI Grok voice realtime behind the role-agnostic ``Backend`` contract."""

    _SESSION_CLS: ClassVar[type[OpenAIRealtimeSession]] = GrokVoiceSession
    _API_KEY_ENV: ClassVar[str] = "XAI_API_KEY"

    def __init__(self, *, config: dict[str, Any]) -> None:
        # xAI defaults; any explicit config value wins. api_key falls back to
        # XAI_API_KEY in the parent (via _API_KEY_ENV).
        merged = {"base_url": XAI_REALTIME_BASE_URL, "voice": DEFAULT_VOICE, **config}
        super().__init__(config=merged)

    @staticmethod
    def _map_event(session: OpenAIRealtimeSession, event: Any) -> list[BackendEvent]:
        """Normalize one xAI event, buffering incremental input transcriptions.

        Defers to ``OpenAIRealtimeBackend._map_event`` for everything except the
        progressive ``input_audio_transcription.completed`` stream, which is
        buffered and flushed as one final input ``TRANSCRIPT`` when the turn
        settles (mirrors ``GrokVoiceAssistantServer``'s deferred transcript).
        """
        event_type = getattr(event, "type", "")

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = (getattr(event, "transcript", "") or "").strip()
            if transcript and isinstance(session, GrokVoiceSession):
                session.pending_input_transcript = transcript  # buffer, don't emit yet
            return []

        # Flush the buffered transcript just before the turn boundary is handled.
        if event_type in ("input_audio_buffer.speech_started", "response.done"):
            return GrokVoiceBackend._flush_pending_input(session) + OpenAIRealtimeBackend._map_event(session, event)

        return OpenAIRealtimeBackend._map_event(session, event)

    @staticmethod
    def _flush_pending_input(session: OpenAIRealtimeSession) -> list[BackendEvent]:
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
