"""NVIDIA Parakeet streaming speech-to-text service implementation.

Follows the same pattern as Pipecat's built-in AssemblyAI STT service.
The subclass only handles server-specific protocol (connection, audio format,
message parsing). All VAD, TTFB metrics, and finalization are handled by the
WebsocketSTTService base class.
"""

import asyncio
import json
import ssl
import time
from collections.abc import AsyncGenerator

import websockets
from loguru import logger
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
from pipecat.services.stt_service import WebsocketSTTService


def current_time_ms():
    return str(int(round(time.time() * 1000)))


class NVidiaWebSocketSTTService(WebsocketSTTService):
    """NVIDIA Parakeet streaming speech-to-text service.

    Provides real-time speech recognition using NVIDIA's Parakeet ASR model
    via WebSocket.

    Server protocol:
    - Audio in:  16-bit PCM, 16kHz, mono (raw bytes)
    - Reset in:  {"type": "reset", "finalize": true}  (triggers final transcript)
    - Ready out: {"type": "ready"}
    - Transcript out: {"type": "transcript", "text": "...", "is_final": true/false}
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8080",
        api_key: str | None = None,
        sample_rate: int = 16000,
        verify: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._api_key = api_key
        self._verify = verify
        self._websocket = None
        self._receive_task: asyncio.Task | None = None
        self._ready = False

    def can_generate_metrics(self) -> bool:
        return True

    # -- Lifecycle (matches AssemblyAI pattern exactly) --

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    # -- Audio sending --

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._websocket and self._ready:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
        yield None

    # -- VAD handling (send reset on speech end, like AssemblyAI's ForceEndpoint) --

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._ready:
                self.request_finalize()
                try:
                    await self._websocket.send(json.dumps({"type": "reset", "finalize": True}))
                except Exception as e:
                    logger.error(f"{self} failed to send reset: {e}")
            await self.start_processing_metrics()

    # -- Connection management --

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
        try:
            ssl_context = None
            if self._url.startswith("wss://") and not self._verify:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            extra_headers = {}
            if self._api_key:
                extra_headers["Authorization"] = f"Bearer {self._api_key}"

            self._websocket = await websockets.connect(
                self._url,
                ssl=ssl_context,
                additional_headers=extra_headers or None,
            )
            self._ready = False

            # Wait for ready message from server
            try:
                ready_msg = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                data = json.loads(ready_msg)
                if data.get("type") == "ready":
                    self._ready = True
                    logger.info(f"{self} connected and ready")
                else:
                    logger.warning(f"{self} unexpected initial message: {data}")
                    self._ready = True
            except TimeoutError:
                logger.warning(f"{self} timeout waiting for ready, proceeding")
                self._ready = True

            await self._call_event_handler("on_connected", self)

        except Exception as e:
            logger.error(f"{self} connection failed: {e}")
            raise

    async def _disconnect_websocket(self):
        self._ready = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} error closing websocket: {e}")
            finally:
                self._websocket = None
                await self._call_event_handler("on_disconnected", self)

    # -- Message receiving --

    async def _receive_messages(self):
        if not self._websocket:
            return

        async for message in self._websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "transcript":
                    await self._handle_transcript(data)
                elif msg_type == "ready":
                    self._ready = True
                elif msg_type == "error":
                    logger.error(f"{self} server error: {data.get('message')}")

            except json.JSONDecodeError:
                logger.warning(f"{self} non-JSON message received")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _handle_transcript(self, data: dict):
        text = data.get("text", "")
        is_final = data.get("is_final", False)

        if not text:
            # Empty reset response (ghost turn). Push empty finalized
            # TranscriptionFrame so the aggregator resolves immediately.
            if is_final:
                logger.debug(f"{self} empty final transcript (ghost turn)")
                self.confirm_finalize()
            return

        if is_final:
            self.confirm_finalize()
            await self.push_frame(TranscriptionFrame(text, self._user_id, current_time_ms(), language=None))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(InterimTranscriptionFrame(text, self._user_id, current_time_ms(), language=None))
