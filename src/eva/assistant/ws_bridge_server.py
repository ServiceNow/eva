"""Shared base for assistant servers that bridge EVA to a WebSocket voice backend.

EVA's user simulator dials in over a Twilio-framed WebSocket; a voice backend
speaks its own protocol on its own socket. Every such integration needs the same
middle layer, and getting any part of it wrong corrupts the evaluation rather
than merely degrading audio:

* **Output pacing.** The simulator infers turn boundaries from arrival timing,
  so assistant audio must leave at wall-clock 20 ms / 160-byte cadence.
* **Track alignment.** The two recording buffers form a shared timeline; each
  must be padded to the other's position before it advances, or the mixed WAV
  is skewed and every audio judge metric reads the wrong thing.
* **Recording rate.** Both buffers are written at 24 kHz regardless of the
  backend's wire rates, so inbound mu-law is converted twice: once for the wire,
  once for the recording buffer.

Subclasses supply only a ``VoiceBackendSession`` and a model identifier.

See docs/assistant_server_contract.md for the contract this satisfies. Note that
the doc is stale on shutdown: ``AbstractAssistantServer.stop()`` is a concrete
template method and ``_shutdown()`` is the hook implemented here.
"""

import asyncio
import contextlib
import json
import time
from abc import abstractmethod
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, WebSocket

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_16k,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.bridge_events import (
    AssistantTranscript,
    AudioChunk,
    BackendError,
    SpeechStarted,
    ToolCall,
    TurnDone,
    UserTranscript,
    VoiceBackendSession,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

#: Both recording buffers are written at this rate, whatever the backend uses.
RECORDING_SAMPLE_RATE = 24000

#: mu-law bytes per 20 ms frame at 8 kHz — one Twilio media message.
MULAW_FRAME_BYTES = 160
FRAME_INTERVAL_S = 0.02

#: A model_response latency outside this bound indicates a bookkeeping bug, not a slow model.
MAX_PLAUSIBLE_LATENCY_MS = 30_000

#: mu-law 8 kHz -> PCM16 at the backend's input rate.
_TO_BACKEND: dict[int, Callable[[bytes], bytes]] = {
    16000: mulaw_8k_to_pcm16_16k,
    24000: mulaw_8k_to_pcm16_24k,
}

#: PCM16 at the backend's output rate -> mu-law 8 kHz.
_FROM_BACKEND: dict[int, Callable[[bytes], bytes]] = {
    24000: pcm16_24k_to_mulaw_8k,
}


class WebSocketBridgeAssistantServer(AbstractAssistantServer):
    """Bridges the user simulator's Twilio socket to a ``VoiceBackendSession``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_sample_rate = RECORDING_SAMPLE_RATE

        self._session: VoiceBackendSession | None = None
        self._to_backend: Callable[[bytes], bytes] | None = None
        self._from_backend: Callable[[bytes], bytes] | None = None

        self._stream_sid: str = self.conversation_id
        self._audio_out: asyncio.Queue[bytes] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

        # Turn bookkeeping
        self._user_speech_start_ms: int | None = None
        self._user_speech_stop_ms: int | None = None
        self._user_speaking = False
        self._assistant_speaking = False
        self._turn_first_audio_ms: int | None = None
        self._assistant_text = ""

    # ── Subclass contract ─────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier recorded in the metrics log."""
        ...

    @abstractmethod
    async def _open_backend(self) -> VoiceBackendSession:
        """Open a session with the backend. Raise to abort the conversation."""
        ...

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def ws_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.05)
        self._running = True
        logger.info(f"{type(self).__name__} listening on ws://localhost:{self.port}/ws")

    async def _shutdown(self) -> None:
        self._running = False

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.aclose()
            self._session = None

        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task

    # ── Session ───────────────────────────────────────────────────────

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Run one conversation: simulator socket in, backend socket out."""
        try:
            self._session = await self._open_backend()
        except Exception as e:
            # Loud failure: a silent fallback would yield a scored but meaningless run.
            logger.error(f"Failed to open backend session, aborting conversation: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                await websocket.close()
            return

        self._to_backend = self._converter(_TO_BACKEND, self._session.backend_input_rate, "input")
        self._from_backend = self._converter(_FROM_BACKEND, self._session.backend_output_rate, "output")

        self._tasks = [
            asyncio.create_task(self._forward_user_audio(websocket)),
            asyncio.create_task(self._process_backend_events()),
            asyncio.create_task(self._pace_audio_output(websocket)),
        ]
        done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception() is not None:
                logger.error(f"Bridge task failed: {task.exception()}")

    @staticmethod
    def _converter(table: dict[int, Callable[[bytes], bytes]], rate: int, direction: str) -> Callable[[bytes], bytes]:
        converter = table.get(rate)
        if converter is None:
            raise ValueError(
                f"No mu-law converter for backend {direction} rate {rate} Hz "
                f"(available: {sorted(table)}). Add one to audio_bridge.py."
            )
        return converter

    async def _forward_user_audio(self, websocket: WebSocket) -> None:
        """Simulator -> backend, plus the user recording track."""
        while self._running:
            try:
                raw = await websocket.receive_text()
            except Exception:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = msg.get("event")

            if event == "start":
                self._stream_sid = msg.get("start", {}).get("streamSid", self._stream_sid)
            elif event == "stop":
                break
            elif event == "user_speech_start":
                self._user_speech_start_ms = _as_int_ms(msg.get("timestamp_ms"))
                self._user_speaking = True
            elif event == "user_speech_stop":
                self._user_speech_stop_ms = _as_int_ms(msg.get("timestamp_ms"))
                self._user_speaking = False
            elif event == "media":
                mulaw = parse_twilio_media_message(raw)
                if not mulaw:
                    continue
                await self._session.send_audio(self._to_backend(mulaw))
                self._record_user_audio(mulaw)

    def _record_user_audio(self, mulaw: bytes) -> None:
        """Append to the user track at the recording rate, keeping tracks aligned."""
        pcm = mulaw_8k_to_pcm16_24k(mulaw)
        if not self._assistant_speaking:
            sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
        self.user_audio_buffer.extend(pcm)

    async def _process_backend_events(self) -> None:
        """Backend -> audit log, metrics, recording track, and output queue."""
        async for event in self._session.events():
            if isinstance(event, AudioChunk):
                self._on_audio_chunk(event.pcm)
            elif isinstance(event, AssistantTranscript):
                self._assistant_text = event.text
            elif isinstance(event, UserTranscript):
                timestamp = str(self._user_speech_start_ms or self._now_ms())
                self.audit_log.append_user_input(event.text, timestamp_ms=timestamp)
            elif isinstance(event, ToolCall):
                await self._handle_tool_call(event)
            elif isinstance(event, SpeechStarted):
                self._handle_barge_in()
            elif isinstance(event, TurnDone):
                self._finish_turn(interrupted=False)
            elif isinstance(event, BackendError):
                logger.error(f"Backend error: {event.message}")
                if event.fatal:
                    break

    def _on_audio_chunk(self, pcm: bytes) -> None:
        """Record assistant audio, start the turn if needed, and queue for paced send."""
        if self._turn_first_audio_ms is None:
            self._turn_first_audio_ms = self._now_ms()
            self._fw_log.turn_start(timestamp_ms=self._turn_first_audio_ms)
            self._write_model_response_latency(self._turn_first_audio_ms)

        self._assistant_speaking = True
        if not self._user_speaking:
            sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
        self.assistant_audio_buffer.extend(pcm)
        self._audio_out.put_nowait(self._from_backend(pcm))

    def _write_model_response_latency(self, first_audio_ms: int) -> None:
        """Time from the simulator's user_speech_stop to our first audio byte.

        The simulator's VAD is the contract-specified source. The backend's own
        speech events drive barge-in only; mixing the two skews this metric.
        """
        if self._user_speech_stop_ms is None:
            return  # Model-initiated turn, e.g. the opening greeting.

        latency_ms = first_audio_ms - self._user_speech_stop_ms
        if 0 < latency_ms < MAX_PLAUSIBLE_LATENCY_MS:
            self._metrics_log.write_latency("model_response", latency_ms / 1000, self.model_name)
        else:
            logger.warning(f"Discarding implausible model_response latency: {latency_ms} ms")
        self._user_speech_stop_ms = None

    def _finish_turn(self, interrupted: bool) -> None:
        """Close out the assistant turn. Idempotent — both reply_done and audio_done arrive."""
        if self._turn_first_audio_ms is None and not self._assistant_text:
            return

        text = self._assistant_text
        if interrupted and text:
            text = f"{text} [interrupted]"

        if text:
            timestamp = str(self._turn_first_audio_ms or self._now_ms())
            self.audit_log.append_assistant_output(text, timestamp_ms=timestamp)
            self._fw_log.llm_response(text)
            self._fw_log.s2s_transcript(text)
        self._fw_log.turn_end(was_interrupted=interrupted)

        self._assistant_text = ""
        self._turn_first_audio_ms = None
        self._assistant_speaking = False

    def _handle_barge_in(self) -> None:
        """Drop undelivered assistant audio and close the turn as interrupted."""
        if self._turn_first_audio_ms is None and not self._assistant_text:
            return  # Assistant was not speaking: an ordinary user turn starting.

        dropped = 0
        while not self._audio_out.empty():
            self._audio_out.get_nowait()
            dropped += 1
        if dropped:
            logger.debug(f"Barge-in: dropped {dropped} queued audio frames")
        self._finish_turn(interrupted=True)

    async def _handle_tool_call(self, event: ToolCall) -> None:
        """Execute against the scenario database and relay the result back."""
        try:
            result = await self.execute_tool(event.name, event.arguments)
        except Exception as e:
            logger.error(f"Tool {event.name} raised: {e}", exc_info=True)
            result = {"status": "error", "message": str(e)}
        await self._session.send_tool_result(event.call_id, result)

    async def _pace_audio_output(self, websocket: WebSocket) -> None:
        """Drain the output queue at 20 ms per 160-byte frame.

        Backend audio does not arrive in 160-byte multiples once converted, so
        frames are re-chunked here. Sending faster or slower than real time makes
        the simulator misjudge turn boundaries.
        """
        pending = bytearray()
        next_send = time.monotonic()

        while self._running:
            chunk = await self._audio_out.get()
            pending.extend(chunk)

            while len(pending) >= MULAW_FRAME_BYTES:
                frame = bytes(pending[:MULAW_FRAME_BYTES])
                del pending[:MULAW_FRAME_BYTES]
                try:
                    await websocket.send_text(create_twilio_media_message(self._stream_sid, frame))
                except Exception:
                    return

                next_send += FRAME_INTERVAL_S
                sleep_s = next_send - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                else:
                    # Fell behind (or the queue was idle): resync rather than burst.
                    next_send = time.monotonic()

    @staticmethod
    def _now_ms() -> int:
        """Wall-clock milliseconds. Overridden in tests."""
        return int(time.time() * 1000)


def _as_int_ms(value: object) -> int | None:
    """Coerce a simulator timestamp to int milliseconds, tolerating str or float."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
