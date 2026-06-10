"""NVIDIA Parakeet streaming speech-to-text service implementation.

Audio gating strategy — bot-speaking gate:
  - The audio gate is OPEN by default.  Audio flows to Parakeet whenever the
    bot is not speaking.
  - The gate CLOSES on BotStartedSpeakingFrame.  Any buffered transcript
    parts are discarded so that stale Parakeet completions from the
    inter-turn silence period do not bleed into the next user turn.
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
import queue
import ssl
import time
from collections import deque
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
from pipecat.services.stt_service import STTService, WebsocketSTTService

from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Seconds after VAD stop to wait for a `completed` before force-finalizing.
_FINALIZE_TIMEOUT_SECS = 3.0

# Pre-roll buffer depth for NVidiaRivaSTTService.
# Silero VAD fires ~200–500 ms after speech starts, so the gRPC stream would
# otherwise miss the opening words.  Keeping the last N chunks (at typical
# 20 ms/chunk ≈ 1 s) and prepending them on utterance start fixes the truncation.
_PRE_ROLL_CHUNKS = 50  # 50 × 20 ms ≈ 1 s

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

    # -- Frame handling (bot-speaking gate + VAD finalization) --

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # --- Bot-speaking gate ---
        if isinstance(frame, BotStartedSpeakingFrame):
            self._audio_gate_open = False
            # Discard any stale transcript parts so old Parakeet completions
            # from the inter-turn silence period don't bleed into the next turn.
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
        """Auto-finalize using Parakeet's transcript when VAD missed the speech.

        Because VAD never fired, the downstream LLMUserAggregator has no
        active user turn.  We push synthetic VAD start/stop frames so the
        aggregator sees a proper turn lifecycle and triggers the LLM.
        """
        await asyncio.sleep(_FALLBACK_FINALIZE_SECS)
        if self._transcript_parts and not self._finalize_requested:
            logger.warning(
                f"{self} VAD miss — fallback finalizing with Parakeet transcript after {_FALLBACK_FINALIZE_SECS}s"
            )
            # Push synthetic VAD start so the aggregator opens a user turn.
            await self.push_frame(VADUserStartedSpeakingFrame())

            self._finalize_requested = True
            self.request_finalize()
            await self.start_processing_metrics()
            await self._emit_final_transcript()

            # Push synthetic VAD stop so the aggregator closes the turn
            # and triggers the LLM.
            await self.push_frame(VADUserStoppedSpeakingFrame())

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


class NVidiaRivaSTTService(STTService):
    """NVIDIA Parakeet RNNT streaming STT via Riva gRPC.

    Each user utterance (VAD start → VAD stop) is a single stateless gRPC
    streaming call, eliminating server-side VAD state bleed between turns.

    Audio gate strategy mirrors NVidiaWebSocketSTTService: the gate closes
    while the bot is speaking and reopens when it stops.
    """

    def __init__(
        self,
        *,
        server: str,
        model_name: str,
        language_code: str = "multi",
        use_ssl: bool = False,
        api_key: str | None = None,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=None, language=None),
            **kwargs,
        )
        self._server = server
        self._model_name = model_name
        self._language_code = language_code
        self._use_ssl = use_ssl
        self._api_key = api_key
        self._asr_service = None
        self._riva = None
        self._audio_q: queue.Queue | None = None
        self._pending_transcript: str | None = None
        self._audio_gate_open = True
        self._finalize_requested = False
        self._finalize_timeout_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Circular buffer of recent audio chunks captured while the gate is open.
        # Drained into new gRPC streams so speech before VAD fires isn't lost.
        self._pre_roll_buffer: deque = deque(maxlen=_PRE_ROLL_CHUNKS)

    def can_generate_metrics(self) -> bool:
        return True

    # -- Lifecycle --

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._loop = asyncio.get_running_loop()
        try:
            import riva.client as _riva
        except ImportError as e:
            raise RuntimeError("nvidia-riva-client not installed. Run: pip install nvidia-riva-client") from e

        try:
            self._riva = _riva
            metadata = [["authorization", f"Bearer {self._api_key}"]] if self._api_key else None
            auth = _riva.Auth(use_ssl=self._use_ssl, uri=self._server, metadata_args=metadata)
            self._asr_service = _riva.ASRService(auth)
            logger.info(f"{self} Riva gRPC ready at {self._server} (model={self._model_name})")
        except Exception as e:
            logger.error(f"{self} failed to initialize Riva gRPC client: {e}")
            raise

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._cancel_finalize_timeout()
        self._signal_audio_done()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._cancel_finalize_timeout()
        self._signal_audio_done()

    # -- Audio processing --

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Feed audio chunk into the active gRPC stream."""
        if self._audio_gate_open:
            self._pre_roll_buffer.append(audio)
            if self._audio_q is not None:
                self._audio_q.put_nowait(audio)
        yield None

    # -- Frame handling --

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._audio_gate_open = False
            self._pending_transcript = None
            self._signal_audio_done()
            self._pre_roll_buffer.clear()  # stale inter-turn audio; next turn starts fresh
            logger.debug(f"{self} audio gate CLOSED (bot speaking)")

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._audio_gate_open = True
            logger.debug(f"{self} audio gate OPEN (bot stopped)")

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._cancel_finalize_timeout()
            self._pending_transcript = None
            self._start_utterance()

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._finalize_requested = True
            self.request_finalize()
            await self.start_processing_metrics()
            self._signal_audio_done()
            if self._pending_transcript is not None:
                # gRPC already finished before VAD stop — use cached result.
                text = self._pending_transcript
                self._pending_transcript = None
                await self._emit_final_transcript(text)
            else:
                self._start_finalize_timeout()

    # -- Utterance lifecycle --

    def _start_utterance(self):
        """Start a fresh gRPC streaming call for this utterance."""
        self._signal_audio_done()  # clean up any in-flight utterance
        audio_q: queue.Queue = queue.Queue()
        # Prepend pre-VAD audio so the gRPC stream captures speech that was
        # spoken before Silero VAD fired (typically 200–500 ms of context).
        for chunk in self._pre_roll_buffer:
            audio_q.put_nowait(chunk)
        self._audio_q = audio_q
        loop = self._loop

        async def _run_in_executor():
            assert loop is not None
            await loop.run_in_executor(None, self._grpc_stream_thread, audio_q, loop)

        self.create_task(_run_in_executor())

    # Trailing silence injected after VAD stop so the server's internal Silero
    # VAD has enough context to detect end-of-speech and emit is_final=True.
    _TRAILING_SILENCE_MS = 800

    def _signal_audio_done(self):
        """Inject trailing silence then send None sentinel to stop the audio iterator.

        The server (Parakeet RNNT) uses Silero VAD internally to decide when to
        emit is_final=True.  Silero needs trailing silence — without it the stream
        closes before the server VAD fires and we never get a final result.
        """
        if self._audio_q is not None:
            try:
                # 16-bit mono: 2 bytes per sample
                silence_bytes = bytes(int(self._sample_rate * self._TRAILING_SILENCE_MS / 1000) * 2)
                chunk_size = int(self._sample_rate * 0.02) * 2  # 20 ms chunks
                for i in range(0, len(silence_bytes), chunk_size):
                    self._audio_q.put_nowait(silence_bytes[i : i + chunk_size])
            except Exception:
                pass
            try:
                self._audio_q.put_nowait(None)
            except Exception:
                pass
            self._audio_q = None

    def _grpc_stream_thread(self, audio_q: queue.Queue, loop: asyncio.AbstractEventLoop):
        """Synchronous gRPC streaming call — runs in a thread-pool executor."""
        try:
            config = self._riva.StreamingRecognitionConfig(
                config=self._riva.RecognitionConfig(
                    encoding=self._riva.AudioEncoding.LINEAR_PCM,
                    sample_rate_hertz=self._sample_rate,
                    language_code=self._language_code,
                    model=self._model_name,
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                ),
                interim_results=True,
            )

            def _audio_iter():
                while True:
                    chunk = audio_q.get()
                    if chunk is None:
                        return
                    yield chunk

            transcript_parts: list[str] = []
            last_text: str = ""

            for response in self._asr_service.streaming_response_generator(
                audio_chunks=_audio_iter(),
                streaming_config=config,
            ):
                for result in response.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if not text:
                        continue
                    if result.is_final:
                        transcript_parts.append(text)
                    last_text = text
                    asyncio.run_coroutine_threadsafe(self._push_interim(text), loop)

            # Prefer is_final parts; fall back to the last RNNT hypothesis when
            # the server's Silero VAD never fires before the stream closes.
            final_text = " ".join(transcript_parts) if transcript_parts else last_text
            asyncio.run_coroutine_threadsafe(self._on_grpc_done(final_text), loop)

        except Exception as e:
            logger.error(f"{self} gRPC error: {e}")
            asyncio.run_coroutine_threadsafe(self._on_grpc_error(e), loop)

    # -- Async callbacks from gRPC thread --

    async def _push_interim(self, text: str):
        await self.push_frame(InterimTranscriptionFrame(text, self._user_id, current_time_ms(), language=None))

    async def _on_grpc_done(self, text: str):
        """Called when the gRPC stream finishes (all audio consumed)."""
        await self._cancel_finalize_timeout()
        if self._finalize_requested:
            if text:
                await self._emit_final_transcript(text)
            else:
                logger.debug(f"{self} ghost turn (empty gRPC result)")
                self._finalize_requested = False
                self.confirm_finalize()
                await self.stop_processing_metrics()
        else:
            # gRPC finished before external VAD fired (rare).
            # Cache the transcript; VADUserStoppedSpeakingFrame will pick it up.
            logger.debug(f"{self} gRPC done before VAD stop: {text!r}")
            if text:
                self._pending_transcript = text
                await self._push_interim(text)

    async def _on_grpc_error(self, e: Exception):
        if self._finalize_requested:
            logger.error(f"{self} gRPC error during finalization: {e}")
            self._finalize_requested = False
            self.confirm_finalize()
            await self.stop_processing_metrics()

    # -- Finalize timeout --

    def _start_finalize_timeout(self):
        if self._finalize_timeout_task:
            self._finalize_timeout_task.cancel()
        self._finalize_timeout_task = self.create_task(self._finalize_timeout_handler())

    async def _cancel_finalize_timeout(self):
        if self._finalize_timeout_task:
            await self.cancel_task(self._finalize_timeout_task)
            self._finalize_timeout_task = None

    async def _finalize_timeout_handler(self):
        await asyncio.sleep(_FINALIZE_TIMEOUT_SECS)
        if self._finalize_requested:
            logger.warning(f"{self} finalize timeout after {_FINALIZE_TIMEOUT_SECS}s — force-finalizing")
            self._finalize_requested = False
            self.confirm_finalize()
            await self.stop_processing_metrics()

    async def _emit_final_transcript(self, text: str):
        self._finalize_requested = False
        await self._cancel_finalize_timeout()
        logger.info(f"{self} final transcript: {text}")
        self.confirm_finalize()
        await self.push_frame(TranscriptionFrame(text, self._user_id, current_time_ms(), language=None, finalized=True))
        await self.stop_processing_metrics()
