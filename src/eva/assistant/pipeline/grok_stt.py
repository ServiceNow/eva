"""xAI Grok streaming speech-to-text via the dedicated STT WebSocket endpoint.

Connects to ``wss://api.x.ai/v1/stt`` and streams raw PCM audio.  The server
returns three transcript states distinguished by ``is_final`` / ``speech_final``:

  * interim  (is_final=False) — text may still change
  * chunk-final (is_final=True, speech_final=False) — ~3 s of locked text
  * utterance-final (is_final=True, speech_final=True) — speaker paused

Interim and chunk-final events are emitted as ``InterimTranscriptionFrame``;
utterance-final events are emitted as ``TranscriptionFrame``.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from eva.utils.logging import get_logger

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    raise Exception(
        'websockets package is required for Grok STT. Install it with: pip install "pipecat-ai[websocket]"'
    ) from e

logger = get_logger(__name__)

XAI_STT_WS_URL = "wss://api.x.ai/v1/stt"


class GrokSTTService(WebsocketSTTService):
    """xAI Grok streaming speech-to-text service.

    Streams raw PCM audio over a WebSocket to xAI's dedicated STT endpoint
    and emits pipecat transcription frames from the returned events.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = XAI_STT_WS_URL,
        sample_rate: int = 16000,
        encoding: str = "pcm",
        interim_results: bool = True,
        endpointing: int = 10,
        language: str | None = None,
        **kwargs: Any,
    ):
        """Initialise the Grok STT service.

        Args:
            api_key: xAI API key.
            base_url: WebSocket URL for the xAI STT endpoint.
            sample_rate: Audio sample rate in Hz (16 000 recommended).
            encoding: Audio encoding — ``pcm``, ``mulaw``, or ``alaw``.
            interim_results: Emit partial transcripts (~every 500 ms).
            endpointing: Silence duration (ms) before utterance-final (0–5000).
            language: Language code for text formatting (e.g. ``en``).
            **kwargs: Passed through to ``WebsocketSTTService``.
        """
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=None, language=language or "en"),
            **kwargs,
        )
        self._api_key = api_key
        self._base_url = base_url
        self._encoding = encoding
        self._interim_results = interim_results
        self._endpointing = endpointing
        self._language = language

        self._connected = False
        self._receive_task = None

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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send raw audio bytes to xAI STT."""
        if self._websocket and self._websocket.state is State.OPEN:
            await self._websocket.send(audio)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.start_processing_metrics()

    # ── WebSocket lifecycle ──────────────────────────────────────────

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if not self._connected:
            return
        try:
            # Signal end-of-audio before closing
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "audio.done"}))
        except Exception:
            pass
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        if self._websocket and self._websocket.state is State.OPEN:
            return

        url = self._build_ws_url()
        logger.debug(f"Connecting to Grok STT: {url}")

        self._websocket = await websocket_connect(
            uri=url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
        )
        self._connected = True

        # Wait for transcript.created before sending audio
        raw = await self._websocket.recv()
        msg = json.loads(raw)
        if msg.get("type") != "transcript.created":
            logger.warning(f"Expected transcript.created, got: {msg.get('type')}")

        logger.debug("Grok STT WebSocket connected and ready")

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Grok STT WebSocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"Error closing Grok STT WebSocket: {e}")
        finally:
            self._websocket = None
            self._connected = False

    async def _receive_messages(self):
        async for message in self._websocket:
            try:
                data = json.loads(message)
                await self._handle_event(data)
            except json.JSONDecodeError:
                logger.warning("Grok STT: received non-JSON message")

    # ── Event handling ───────────────────────────────────────────────

    async def _handle_event(self, data: dict[str, Any]) -> None:
        event_type = data.get("type", "")

        if event_type == "transcript.partial":
            text = (data.get("text") or "").strip()
            if not text:
                return

            is_final = data.get("is_final", False)
            speech_final = data.get("speech_final", False)

            if is_final and speech_final:
                # Utterance-final: speaker paused, emit as final transcript
                logger.debug(f"Grok STT final: {text[:60]}...")
                await self.push_frame(TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN, data))
                await self.stop_processing_metrics()
            else:
                # Interim or chunk-final: text may still evolve
                await self.push_frame(
                    InterimTranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN, data)
                )

        elif event_type == "transcript.done":
            text = (data.get("text") or "").strip()
            if text:
                logger.debug(f"Grok STT done: {text[:60]}...")
                await self.push_frame(TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN, data))
                await self.stop_processing_metrics()

        elif event_type == "error":
            logger.error(f"Grok STT error: {data.get('message')}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with query parameters."""
        params = {
            "sample_rate": self.sample_rate,
            "encoding": self._encoding,
            "interim_results": str(self._interim_results).lower(),
            "endpointing": self._endpointing,
        }
        if self._language:
            params["language"] = self._language

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._base_url}?{query}"
