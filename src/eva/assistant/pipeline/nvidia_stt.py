"""NVIDIA Parakeet streaming speech-to-text service implementation.

Audio gating strategy — bot-speaking gate with buffer clearing:
  - The audio gate is OPEN by default.  Audio flows to Parakeet whenever the
    bot is not speaking.
  - The gate CLOSES on BotStartedSpeakingFrame.  At the same time we send
    ``input_audio_buffer.clear`` to Parakeet and discard any buffered
    transcript parts so that stale audio from the previous inter-turn gap
    does not bleed into the next user turn.
  - The gate OPENS on BotStoppedSpeakingFrame, resuming normal audio flow.
  - A keepalive sends silent audio during long bot-speech turns to prevent
    the Parakeet WebSocket from closing.

Finalization (VAD-primary, Parakeet-fallback):
  - When VAD fires stop, finalize immediately or wait for the next
    ``completed`` event (primary path).
  - If Parakeet emits a non-empty ``completed`` and VAD has NOT fired, a
    fallback timer starts.  If VAD still hasn't fired when the timer
    expires, we auto-finalize using Parakeet's transcript — this handles
    the case where Silero VAD misses a short utterance.
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
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService

from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Seconds after VAD stop to wait for a `completed` before force-finalizing.
_FINALIZE_TIMEOUT_SECS = 3.0

# Seconds after a Parakeet `completed` (with no VAD) before auto-finalizing.
# Gives VAD a chance to catch up; if it doesn't, Parakeet's own sentence
# detection serves as the fallback signal.
_FALLBACK_FINALIZE_SECS = 1.5


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
            # Send a silent keepalive every 10s after 15s of no audio, so the
            # Parakeet WebSocket doesn't close during long bot-speech turns.
            keepalive_timeout=15.0,
            keepalive_interval=10.0,
            **kwargs,
        )
        self._url = url
        self._api_key = api_key
        self._verify = verify
        self._asr_model = None
        self._websocket = None
        self._receive_task: asyncio.Task | None = None
        self._ready = False
        # Gate starts OPEN — audio flows to Parakeet by default.
        # Only closed while the bot is speaking.
        self._audio_gate_open = True
        self._finalize_requested = False
        self._finalize_timeout_task: asyncio.Task | None = None
        self._fallback_finalize_task: asyncio.Task | None = None
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

    # -- Audio processing --

    _audio_chunk_count: int = 0

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Override base class to only reset keepalive timer when actually sending.

        The base STTService.process_audio_frame unconditionally resets
        ``_last_audio_time`` on every audio frame — including silence during
        bot speech.  This prevents the keepalive from ever firing, so the
        Parakeet WebSocket dies during long bot turns.

        When the gate is closed (bot speaking) we skip the base-class call
        entirely so the keepalive timer keeps ticking.
        """
        if self._muted:
            return

        if self._audio_gate_open:
            # Gate open — let the base class update _last_audio_time
            # and call run_stt normally.
            await super().process_audio_frame(frame, direction)
        # Gate closed (bot speaking) — don't touch _last_audio_time so the
        # keepalive timer keeps ticking.  Audio is intentionally discarded.

    async def _send_audio(self, audio: bytes):
        """Send a single audio chunk to Parakeet (append + commit)."""
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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._websocket or not self._ready:
            if not self._ready:
                logger.warning(f"{self} audio dropped — not ready")
            yield None
            return

        await self._send_audio(audio)
        yield None

    # -- Keepalive --

    async def _send_keepalive(self, silence: bytes):
        """Wrap silent PCM in Parakeet's append+commit protocol."""
        logger.debug(f"{self} sending keepalive silence ({len(silence)} bytes)")
        await self._send_audio(silence)

    # -- Parakeet buffer management --

    async def _clear_parakeet_buffer(self):
        """Send input_audio_buffer.clear to flush stale audio in Parakeet.

        Called when the bot starts speaking so that any accumulated inter-turn
        silence doesn't produce stale ``completed`` events on the next turn.
        """
        if self._websocket and self._ready:
            try:
                await self._websocket.send(json.dumps({"type": "input_audio_buffer.clear"}))
                logger.debug(f"{self} cleared Parakeet audio buffer")
            except Exception as e:
                logger.error(f"{self} failed to clear audio buffer: {e}")

    # -- Frame handling (bot-speaking gate + VAD finalization) --

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # --- Bot-speaking gate ---
        if isinstance(frame, BotStartedSpeakingFrame):
            self._audio_gate_open = False
            # Flush stale audio from Parakeet so old transcripts don't bleed
            # into the next user turn.
            await self._clear_parakeet_buffer()
            self._transcript_parts.clear()
            await self._cancel_fallback_finalize()
            logger.debug(f"{self} audio gate CLOSED (bot speaking)")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._audio_gate_open = True
            logger.debug(f"{self} audio gate OPEN (bot stopped)")

        # --- VAD-based finalization (primary path) ---
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # VAD detected speech — cancel any fallback timer since VAD is
            # now in control of finalization.
            await self._cancel_fallback_finalize()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._cancel_fallback_finalize()
            self._finalize_requested = True
            self.request_finalize()
            await self.start_processing_metrics()
            if self._transcript_parts:
                await self._emit_final_transcript()
            else:
                # Start a safety timeout — if Parakeet doesn't send `completed`
                # within a few seconds, force-finalize.
                self._start_finalize_timeout()

    # -- Finalize timeout (VAD fired but no completed from Parakeet) --

    def _start_finalize_timeout(self):
        """Start (or restart) the finalize safety timeout."""
        if self._finalize_timeout_task:
            self._finalize_timeout_task.cancel()
        self._finalize_timeout_task = self.create_task(self._finalize_timeout_handler())

    async def _cancel_finalize_timeout(self):
        """Cancel any pending finalize timeout."""
        if self._finalize_timeout_task:
            await self.cancel_task(self._finalize_timeout_task)
            self._finalize_timeout_task = None

    async def _finalize_timeout_handler(self):
        """Force-finalize after trailing silence timeout."""
        await asyncio.sleep(_FINALIZE_TIMEOUT_SECS)
        if self._finalize_requested:
            logger.warning(f"{self} finalize timeout after {_FINALIZE_TIMEOUT_SECS}s — force-finalizing")
            if self._transcript_parts:
                await self._emit_final_transcript()
            else:
                # Ghost turn — no transcript arrived.
                self._finalize_requested = False
                self.confirm_finalize()

    # -- Fallback finalize (Parakeet completed but VAD never fired) --

    def _start_fallback_finalize(self):
        """Start a fallback timer to auto-finalize if VAD doesn't fire."""
        if self._fallback_finalize_task:
            self._fallback_finalize_task.cancel()
        self._fallback_finalize_task = self.create_task(self._fallback_finalize_handler())

    async def _cancel_fallback_finalize(self):
        """Cancel the fallback finalize timer."""
        if self._fallback_finalize_task:
            await self.cancel_task(self._fallback_finalize_task)
            self._fallback_finalize_task = None

    async def _fallback_finalize_handler(self):
        """Auto-finalize using Parakeet's transcript when VAD missed the speech."""
        await asyncio.sleep(_FALLBACK_FINALIZE_SECS)
        if self._transcript_parts and not self._finalize_requested:
            logger.warning(
                f"{self} VAD miss — fallback finalizing with Parakeet transcript after {_FALLBACK_FINALIZE_SECS}s"
            )
            self._finalize_requested = True
            self.request_finalize()
            await self.start_processing_metrics()
            await self._emit_final_transcript()

    # -- Connection management --

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        await self._cancel_finalize_timeout()
        await self._cancel_fallback_finalize()

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
                # VAD already fired — finalize immediately.
                await self._emit_final_transcript()
            else:
                # VAD hasn't fired yet.  Push as interim and start the
                # fallback timer so we auto-finalize if VAD never fires.
                logger.debug(f"{self} buffered (no VAD yet): {transcript}")
                await self.push_frame(
                    InterimTranscriptionFrame(transcript, self._user_id, current_time_ms(), language=None)
                )
                self._start_fallback_finalize()
        elif self._finalize_requested:
            # Empty completed after VAD fired (silence audio).
            if self._transcript_parts:
                await self._emit_final_transcript()
            else:
                logger.debug(f"{self} ghost turn (empty completed)")
                self._finalize_requested = False
                await self._cancel_finalize_timeout()
                self.confirm_finalize()

    async def _emit_final_transcript(self):
        """Flush accumulated transcript parts and emit a finalized TranscriptionFrame."""
        full_transcript = " ".join(self._transcript_parts)
        self._transcript_parts = []
        self._finalize_requested = False
        await self._cancel_finalize_timeout()
        await self._cancel_fallback_finalize()
        logger.info(f"{self} final transcript: {full_transcript}")
        self.confirm_finalize()
        await self.push_frame(
            TranscriptionFrame(full_transcript, self._user_id, current_time_ms(), language=None, finalized=True)
        )
        await self.stop_processing_metrics()
