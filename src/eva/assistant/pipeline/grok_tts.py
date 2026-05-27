"""xAI Grok streaming text-to-speech via the dedicated TTS WebSocket endpoint.

Connects to ``wss://api.x.ai/v1/tts`` and streams text via the
``text.delta`` / ``text.done`` protocol.  Audio arrives as base64-encoded
PCM chunks in ``audio.delta`` events, followed by ``audio.done`` when the
utterance is complete.

Barge-in is supported: ``on_audio_context_interrupted`` sends ``text.clear``
and waits for the ``audio.clear`` confirmation before resuming.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from eva.utils.logging import get_logger

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    raise Exception(
        'websockets package is required for Grok TTS. Install it with: pip install "pipecat-ai[websocket]"'
    ) from e

logger = get_logger(__name__)

XAI_TTS_WS_URL = "wss://api.x.ai/v1/tts"


@dataclass
class GrokTTSSettings(TTSSettings):
    """Settings for GrokTTSService.

    URL-level params (voice, language, speed, codec, sample_rate, etc.)
    require a reconnect when changed at runtime.
    """

    speed: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    codec: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    bit_rate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    optimize_streaming_latency: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GrokTTSService(WebsocketTTSService):
    """xAI Grok streaming text-to-speech service.

    Streams text over a persistent WebSocket and receives base64-encoded
    PCM audio chunks.  The connection is reused across utterances to avoid
    repeated handshake latency (~600 ms).
    """

    Settings = GrokTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "eve",
        language: str = "en",
        base_url: str = XAI_TTS_WS_URL,
        sample_rate: int = 24000,
        speed: float = 1.0,
        codec: str = "pcm",
        bit_rate: int = 128000,
        optimize_streaming_latency: int = 0,
        text_normalization: bool = False,
        settings: GrokTTSSettings | None = None,
        **kwargs: Any,
    ):
        """Initialise the Grok TTS service.

        Args:
            api_key: xAI API key.
            voice: Voice name — ``ara``, ``eve``, ``leo``, ``rex``, or ``sal``.
            language: BCP-47 language code (e.g. ``en``, ``zh``, ``pt-BR``) or ``auto``.
            base_url: WebSocket URL for the xAI TTS endpoint.
            sample_rate: Output sample rate in Hz.
            speed: Speech speed multiplier (0.7–1.5).
            codec: Audio codec — ``pcm``, ``mp3``, ``wav``, ``mulaw``, or ``alaw``.
            bit_rate: MP3 bit rate (only used when codec is ``mp3``).
            optimize_streaming_latency: Latency optimisation level (0–2).
            text_normalization: Whether to normalise text before synthesis.
            settings: Runtime-updatable settings (takes precedence over init args).
            **kwargs: Passed through to ``WebsocketTTSService``.
        """
        default_settings = self.Settings(
            voice=voice,
            language=language,
            speed=speed,
            codec=codec,
            bit_rate=bit_rate,
            optimize_streaming_latency=optimize_streaming_latency,
            text_normalization=text_normalization,
            model=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=sample_rate, settings=default_settings, **kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._receive_task = None
        self._clear_event = asyncio.Event()

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    # ── TTS generation ───────────────────────────────────────────────

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Send text to xAI TTS and yield control frames.

        Audio arrives asynchronously via ``_receive_messages`` and is
        appended to the audio context there — this method only sends the
        text and yields the bookend frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.audio_context_available(context_id):
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)

                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield TTSStoppedFrame(context_id=context_id)
                yield ErrorFrame(error=f"Grok TTS error: {e}")
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Grok TTS error: {e}")

    async def _send_text(self, text: str) -> None:
        """Send text.delta + text.done to the WebSocket."""
        if not self._websocket:
            return
        # xAI caps individual deltas at 15,000 chars
        max_chunk = 15000
        for i in range(0, len(text), max_chunk):
            chunk = text[i : i + max_chunk]
            await self._websocket.send(json.dumps({"type": "text.delta", "delta": chunk}))
        await self._websocket.send(json.dumps({"type": "text.done"}))

    # ── Interruption handling ────────────────────────────────────────

    async def on_audio_context_interrupted(self, context_id: str):
        """Send text.clear on barge-in and wait for audio.clear confirmation."""
        if self._websocket and self._websocket.state is State.OPEN:
            self._clear_event.clear()
            try:
                await self._websocket.send(json.dumps({"type": "text.clear"}))
                await asyncio.wait_for(self._clear_event.wait(), timeout=2.0)
            except TimeoutError:
                logger.warning("Timed out waiting for audio.clear from Grok TTS")
            except Exception as e:
                logger.error(f"Error sending text.clear: {e}")
        await super().on_audio_context_interrupted(context_id)

    # ── WebSocket lifecycle ──────────────────────────────────────────

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        if self._websocket and self._websocket.state is State.OPEN:
            return

        url = self._build_ws_url()
        logger.debug(f"Connecting to Grok TTS: {url}")

        self._websocket = await websocket_connect(
            uri=url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
        )
        await self._call_event_handler("on_connected")
        logger.debug("Grok TTS WebSocket connected")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Grok TTS")
                await self._websocket.close()
                logger.debug("Disconnected from Grok TTS")
        except Exception as e:
            logger.error(f"Error closing Grok TTS WebSocket: {e}")
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    # ── Receive loop ─────────────────────────────────────────────────

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from xAI TTS."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Grok TTS: received non-JSON message")
                continue

            event_type = msg.get("type", "")
            context_id = self.get_active_audio_context_id()

            if event_type == "audio.delta":
                audio_b64 = msg.get("delta", "")
                if audio_b64 and context_id and self.audio_context_available(context_id):
                    audio = base64.b64decode(audio_b64)
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
                    await self.append_to_audio_context(context_id, frame)
                    await self.stop_ttfb_metrics()

            elif event_type == "audio.done":
                if context_id and self.audio_context_available(context_id):
                    await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
                    await self.remove_audio_context(context_id)

            elif event_type == "audio.clear":
                self._clear_event.set()

            elif event_type == "error":
                logger.error(f"Grok TTS error: {msg.get('message')}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Grok TTS WebSocket not connected")

    # ── Settings updates ─────────────────────────────────────────────

    async def _update_settings(self, delta: GrokTTSSettings) -> dict[str, Any]:  # type: ignore[override]
        """Apply settings delta; reconnect if URL-level params changed."""
        changed = await super()._update_settings(delta)

        # URL-level params require a reconnect
        url_params = {
            "voice",
            "language",
            "speed",
            "codec",
            "bit_rate",
            "sample_rate",
            "optimize_streaming_latency",
            "text_normalization",
        }
        if changed and url_params & changed.keys():
            await self._disconnect()
            await self._connect()

        return changed

    # ── Helpers ──────────────────────────────────────────────────────

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with query parameters from settings."""
        s = self._settings
        params: dict[str, Any] = {
            "voice": s.voice or "eve",
            "language": s.language or "en",
            "codec": s.codec if s.codec is not NOT_GIVEN else "pcm",
            "sample_rate": self.sample_rate,
        }

        if s.speed is not NOT_GIVEN:
            params["speed"] = s.speed
        if s.bit_rate is not NOT_GIVEN:
            params["bit_rate"] = s.bit_rate
        if s.optimize_streaming_latency is not NOT_GIVEN:
            params["optimize_streaming_latency"] = s.optimize_streaming_latency
        if s.text_normalization is not NOT_GIVEN:
            params["text_normalization"] = str(s.text_normalization).lower()

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._base_url}?{query}"
