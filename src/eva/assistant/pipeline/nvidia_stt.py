"""NVIDIA Parakeet streaming speech-to-text service implementation.

The server streams sentence-level `completed` transcription events as the user
speaks. Finalization is driven entirely by Pipecat VAD:

- While VAD is active, incoming `completed` events are accumulated in
  `_transcript_parts` and forwarded as InterimTranscriptionFrame.
- When VAD fires (VADUserStoppedSpeakingFrame), we finalize immediately:
  - If the buffer is already non-empty (server was ahead of VAD), flush now.
  - Otherwise set `_finalize_requested` and emit on the next `completed`.
"""

import asyncio
import base64
import json
import ssl
import time
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import httpx
import websockets
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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


def current_time_ms():
    return str(int(round(time.time() * 1000)))


class NVidiaWebSocketSTTService(WebsocketSTTService):
    """NVIDIA Parakeet streaming speech-to-text service.

    Provides real-time speech recognition using NVIDIA's Parakeet ASR model
    via WebSocket.

    Server protocol (OpenAI Realtime API):
    - Audio in:  {"type": "input_audio_buffer.append", "audio": "<base64 PCM16 16kHz>"}
    - Commit in: {"type": "input_audio_buffer.commit"}
    - Ready out: {"type": "conversation.created"}
    - Transcript out: {"type": "conversation.item.input_audio_transcription.completed", ...}
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8080",
        api_key: str | None = None,
        sample_rate: int = 16000,
        verify: bool = True,
        model: str | None = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=None, language=None),
            **kwargs,
        )
        self._url = url
        self._api_key = api_key
        self._verify = verify
        self._asr_model = None
        self._websocket = None
        self._receive_task: asyncio.Task | None = None
        self._ready = False
        self._bot_speaking = False
        self._finalize_requested = False
        self._transcript_parts: list[str] = []

    def can_generate_metrics(self) -> bool:
        return True

    # -- Lifecycle --

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

    _audio_chunk_count: int = 0

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._websocket and self._ready and not self._bot_speaking:
            try:
                await self._websocket.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii")})
                )
                await self._websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                self._audio_chunk_count += 1
                if self._audio_chunk_count % 50 == 1:
                    logger.debug(f"{self} sent audio chunk #{self._audio_chunk_count} ({len(audio)} bytes)")
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
        elif not self._ready:
            logger.warning(f"{self} audio dropped — not ready")
        yield None

    # -- VAD handling --

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._finalize_requested = True
            self.request_finalize()
            await self.start_processing_metrics()
            # Flush immediately if the server already sent completed chunks
            # before VAD fired — no further completed message will arrive.
            if self._transcript_parts:
                await self._emit_final_transcript()

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

            try:
                logger.info(f"Connecting to {self._url}")
                ready_msg = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                data = json.loads(ready_msg)
                if data.get("type") == "conversation.created":
                    logger.info("Conversation created successfully")
                    await self._configure_session()
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

    async def _initialize_http_session(self) -> dict:
        """Initialize session via HTTP POST to get server defaults (model, sample rate, etc.)."""
        parsed = urlparse(self._url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        http_url = f"{scheme}://{parsed.hostname}"
        if parsed.port:
            http_url += f":{parsed.port}"
        http_url += "/v1/realtime/transcription_sessions"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(verify=self._verify) as client:
            response = await client.post(http_url, headers=headers, json={})
            response.raise_for_status()
            session_data = response.json()
            return session_data

    async def _configure_session(self):
        """Get server defaults via HTTP, then send transcription_session.update over WS."""
        try:
            session_config = await self._initialize_http_session()
        except Exception as e:
            logger.warning(f"{self} HTTP session init failed ({e}), using minimal config")
            session_config = {}

        session_config["input_audio_format"] = "pcm16"

        if self._asr_model:
            session_config.setdefault("input_audio_transcription", {})
            session_config["input_audio_transcription"]["model"] = self._asr_model

        await self._websocket.send(json.dumps({"type": "transcription_session.update", "session": session_config}))

        try:
            response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
            data = json.loads(response)
            if data.get("type") == "transcription_session.updated":
                logger.info(f"{self} session configured: {data.get('session', {})}")
            else:
                logger.warning(f"{self} unexpected session update response: {data}")
        except TimeoutError:
            logger.warning(f"{self} timeout waiting for session update confirmation")

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

                if msg_type == "error":
                    logger.error(f"{self} server error: {data}")
                elif msg_type == "conversation.item.input_audio_transcription.delta":
                    delta = data.get("delta", "")
                    if delta:
                        await self.push_frame(
                            InterimTranscriptionFrame(delta, self._user_id, current_time_ms(), language=None)
                        )
                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    await self._handle_completed(data)

            except json.JSONDecodeError:
                logger.warning(f"{self} non-JSON message received")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _handle_completed(self, data: dict):
        """Handle a server-side sentence completion event."""
        transcript = data.get("transcript", "").strip()

        if transcript:
            self._transcript_parts.append(transcript)
            if self._finalize_requested:
                await self._emit_final_transcript()
            else:
                logger.debug(f"{self} buffered: {transcript}")
                await self.push_frame(
                    InterimTranscriptionFrame(transcript, self._user_id, current_time_ms(), language=None)
                )
        elif self._finalize_requested:
            # Empty completed after VAD fired (silence audio).
            if self._transcript_parts:
                await self._emit_final_transcript()
            else:
                logger.debug(f"{self} ghost turn (empty completed)")
                self._finalize_requested = False
                self.confirm_finalize()

    async def _emit_final_transcript(self):
        """Flush accumulated transcript parts and emit a finalized TranscriptionFrame."""
        full_transcript = " ".join(self._transcript_parts)
        self._transcript_parts = []
        self._finalize_requested = False
        logger.info(f"{self} final transcript: {full_transcript}")
        self.confirm_finalize()
        await self.push_frame(
            TranscriptionFrame(full_transcript, self._user_id, current_time_ms(), language=None, finalized=True)
        )
        await self.stop_processing_metrics()
