"""Grok Voice realtime assistant server (xAI).

xAI's voice realtime API is event-compatible with OpenAI's Realtime API
(per https://docs.x.ai/developers/model-capabilities/audio/voice-agent#openai-realtime-api-compatibility),
so this server subclasses `OpenAIRealtimeAssistantServer` and overrides
only the hooks that differ:

  * `_create_client`                — point AsyncOpenAI at api.x.ai/v1
  * `_default_voice`                — xAI's built-in voices are `eve`/`ara`/`rex`/`sal`/`leo`
  * `_build_session_config`         — xAI doesn't accept the `transcription.model` selector
  * `_on_transcription_completed`   — xAI sends incremental completed events; only flush the final one

The shared audio bridge, event loop, tool round-trip, audit logging,
and latency metrics in `OpenAIRealtimeAssistantServer` are reused as-is.
"""

from typing import Any

from openai import AsyncOpenAI

from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
from eva.utils.logging import get_logger

logger = get_logger(__name__)

XAI_REALTIME_BASE_URL = "https://api.x.ai/v1"


class GrokVoiceAssistantServer(OpenAIRealtimeAssistantServer):
    """Assistant server backed by xAI's Grok voice realtime API."""

    _service_name: str = "Grok Voice"
    _metrics_processor_name: str = "grok_voice"

    def _create_client(self) -> AsyncOpenAI:
        s2s = self.pipeline_config.s2s_params or {}
        api_key = s2s.get("api_key")
        if not api_key:
            raise ValueError(f"API key required for {self._service_name}")
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": s2s.get("base_url", XAI_REALTIME_BASE_URL),
        }
        if websocket_base_url := s2s.get("websocket_base_url"):
            client_kwargs["websocket_base_url"] = websocket_base_url
        return AsyncOpenAI(**client_kwargs)

    def _default_voice(self) -> str:
        return "eve"

    def _build_session_config(self) -> dict[str, Any]:
        """Drop OpenAI's selectable ASR model; xAI transcribes implicitly."""
        config = super()._build_session_config()
        config["audio"]["input"].pop("transcription", None)
        return config

    # ── Deferred transcription (xAI sends incremental completed events) ──

    def _flush_pending_user_transcript(self) -> None:
        """Write the buffered user transcript to the audit log if pending."""
        if self._user_turn and self._user_turn.transcript and not self._user_turn.flushed:
            timestamp_ms = self._user_turn.speech_started_wall_ms or None
            self.audit_log.append_user_input(self._user_turn.transcript, timestamp_ms=timestamp_ms)
            self._user_turn.flushed = True
            logger.debug(f"Flushed deferred user transcript: {self._user_turn.transcript[:60]}...")

    async def _on_transcription_completed(self, event: Any) -> None:
        """Buffer transcription instead of writing immediately.

        xAI fires ``conversation.item.input_audio_transcription.completed``
        multiple times per turn with progressively longer text.  We store
        each update but defer the audit-log write until the turn is done
        (see ``_on_speech_started`` / ``_on_response_done``).
        """
        transcript = getattr(event, "transcript", "") or ""
        transcript = transcript.strip()
        if not transcript:
            return

        user_turn = self._user_turn_for_event(event, create=True)
        if user_turn:
            user_turn.transcript = transcript
            # Do NOT set flushed or write to audit_log yet
        logger.debug(f"Buffered user transcription: {transcript[:60]}...")

    async def _on_speech_started(self, event: Any) -> None:
        """Flush any pending transcript before starting a new turn."""
        self._flush_pending_user_transcript()
        await super()._on_speech_started(event)

    async def _on_response_done(self, event: Any, conn: Any | None = None) -> None:
        """Flush any pending transcript before recording assistant output."""
        self._flush_pending_user_transcript()
        await super()._on_response_done(event, conn)
