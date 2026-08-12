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
  turn, each carrying the *cumulative* text so far rather than a delta, so
  each event is diffed against the last one emitted and only the new suffix
  is surfaced as a ``TRANSCRIPT``.

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
    """OpenAI Realtime session state plus xAI's last cumulative input transcript.

    xAI streams ``input_audio_transcription.completed`` repeatedly, each carrying the
    *cumulative* text so far (it does NOT emit incremental ``.delta`` events like
    OpenAI). Emitting each cumulative event directly repeats text that then double-
    concatenates downstream. ``last_input_transcript`` records the cumulative text we
    last emitted so each ``completed`` surfaces only its new suffix -- a mutually-
    exclusive piece that the postprocessor re-joins with a space.
    """

    last_input_transcript: str = ""


class GrokVoiceBackend(OpenAIRealtimeBackend):
    """xAI Grok voice realtime behind the role-agnostic ``Backend`` contract."""

    _SESSION_CLS: ClassVar[type[OpenAIRealtimeSession]] = GrokVoiceSession
    _API_KEY_ENV: ClassVar[str] = "XAI_API_KEY"

    def __init__(self, *, config: dict[str, Any]) -> None:
        # base_url / voice are xAI defaults any explicit config overrides; api_key falls
        # back to XAI_API_KEY in the parent (via _API_KEY_ENV).
        merged = {"base_url": XAI_REALTIME_BASE_URL, "voice": DEFAULT_VOICE, **config}
        super().__init__(config=merged)

    @staticmethod
    def _map_event(session: OpenAIRealtimeSession, event: Any) -> list[BackendEvent]:
        """Normalize one xAI event, de-duplicating its cumulative input transcripts.

        Defers to ``OpenAIRealtimeBackend._map_event`` for everything except
        ``input_audio_transcription.completed``: xAI re-sends the whole cumulative
        transcript each time, so we emit only the part that extends what we already
        emitted (see ``_emit_input_delta``). xAI sends no ``.delta`` events, so diffing
        the cumulative text is the only incremental signal available.
        """
        if getattr(event, "type", "") == "conversation.item.input_audio_transcription.completed":
            return GrokVoiceBackend._emit_input_delta(session, event)
        return OpenAIRealtimeBackend._map_event(session, event)

    @staticmethod
    def _emit_input_delta(session: OpenAIRealtimeSession, event: Any) -> list[BackendEvent]:
        """Emit only the new suffix of xAI's cumulative input transcript."""
        if not isinstance(session, GrokVoiceSession):
            return []
        transcript = (getattr(event, "transcript", "") or "").strip()
        if not transcript:
            return []

        last = session.last_input_transcript
        if last and transcript.startswith(last):
            piece = transcript[len(last) :].strip()  # extends the current utterance
        else:
            piece = transcript  # a fresh utterance (doesn't extend the last)
        session.last_input_transcript = transcript

        if not piece:
            return []
        return [
            BackendEvent(
                event_type=BackendEventType.TRANSCRIPT,
                transcript=piece,
                metadata={"stream": "input", "final": True},
            )
        ]
