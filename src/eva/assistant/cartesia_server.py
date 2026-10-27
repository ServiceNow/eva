"""Cartesia Managed Agents assistant server.

Bridges the user simulator's Twilio-style WebSocket to a Cartesia managed agent over its WebSocket
API (/v1/agents/websocket/{agent_id}). The agent handles audio and turn-taking; this server moves
audio, runs EVA's tools as client tools, and records the conversation. EVA's system prompt and
greeting are sent with each call as dynamic variables, to agents created by
scripts/provision_cartesia_agents.py. Settings that cannot be set per call are checked against the
run config before the call starts.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import numpy as np
import soxr
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils import cartesia_agents
from eva.utils.audio_utils import create_twilio_media_message, parse_twilio_media_message
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000  # agent PCM in both directions

MULAW_CHUNK_SIZE = 160  # 20 ms at 8 kHz
CHUNK_DURATION_S = 0.02
PCM_CHUNK_SIZE = 640  # 20 ms at 16 kHz

# When the simulator sends nothing for this long, feed real-time silence so the agent can end the
# caller's turn. Never ahead of real time: the simulator sends its own silence in fast bursts.
INPUT_SILENCE_GAP_S = 0.1
_SILENCE_FRAME = b"\x00" * PCM_CHUNK_SIZE


def _now_ms() -> int:
    return int(round(time.time() * 1000))


def _pad_buffer_to_walltime(buffer: bytearray, elapsed_s: float, sample_rate: int) -> None:
    """Pad a PCM16 buffer with silence so its length matches elapsed wall time."""
    target_bytes = int(elapsed_s * sample_rate) * 2
    if len(buffer) < target_bytes:
        buffer.extend(b"\x00" * (target_bytes - len(buffer)))


def _mix_pcm16(track_a: bytes | bytearray, track_b: bytes | bytearray) -> bytes:
    """eva.utils.audio_utils.pcm16_mix, vectorized so teardown does not block the event loop."""
    a = np.frombuffer(track_a, dtype="<i2").astype(np.int32)
    b = np.frombuffer(track_b, dtype="<i2").astype(np.int32)
    n = max(len(a), len(b))
    mixed = np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b)))
    return np.clip(mixed, -32768, 32767).astype("<i2").tobytes()


class _SpeechSegment:
    """One spoken agent turn and the words in it.

    Words are logged in pieces, split at each agent message and tool call and stamped when first
    spoken, so EVA's trace places speech after the tool call that preceded it.
    """

    def __init__(self, start_ms: int):
        self.start_ms = start_ms
        self.text = ""
        self._piece = ""
        self._piece_ms: int | None = start_ms
        self._pieces: list[tuple[str, int]] = []

    def add_text(self, text: str) -> None:
        # Consecutive agent messages arrive unspaced.
        if self.text[-1:] in (".", "!", "?") and text[:1].isalnum():
            self.end_piece()
            text = " " + text
        if self._piece_ms is None:
            self._piece_ms = _now_ms()
        self.text += text
        self._piece += text

    def end_piece(self) -> None:
        if self._piece.strip() and self._piece_ms is not None:
            self._pieces.append((self._piece.strip(), self._piece_ms))
        self._piece, self._piece_ms = "", None

    def take_pieces(self) -> list[tuple[str, int]]:
        """Closed pieces not yet logged, as (text, start_ms)."""
        pieces, self._pieces = self._pieces, []
        return pieces


class CartesiaAssistantServer(AbstractAssistantServer):
    def __init__(
        self,
        current_date_time: str,
        pipeline_config: ModelConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
        language: str = "en",
        turn_end_fallback_time: int | None = None,
    ):
        super().__init__(
            current_date_time=current_date_time,
            pipeline_config=pipeline_config,
            agent=agent,
            agent_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            output_dir=output_dir,
            port=port,
            conversation_id=conversation_id,
            language=language,
        )
        self._audio_sample_rate = _SAMPLE_RATE

        params = self.pipeline_config.s2s_params or {}
        self._params = params
        self._api_key = params["api_key"]
        self._base_url = params.get("base_url", "wss://api.cartesia.ai").rstrip("/")
        self._cartesia_version = params.get("cartesia_version", cartesia_agents.DEFAULT_CARTESIA_VERSION)
        self._turn_end_fallback_time = turn_end_fallback_time
        self._model = params["model"]
        self._cartesia_agent_id = params.get("assistant_agent_id")
        if not self._cartesia_agent_id:
            raise ValueError("Missing Cartesia assistant agent ID in s2s_params")

        self._system_prompt = self._build_system_prompt()
        self._call_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._caller_hung_up: asyncio.Event | None = None

    def notify_conversation_ending(self, reason: str | None = None) -> None:
        """Hang up when the caller does, so the agent stops talking into EVA's teardown."""
        try:
            if self._loop is not None and self._caller_hung_up is not None:
                self._loop.call_soon_threadsafe(self._caller_hung_up.set)
                logger.info(f"Caller hung up ({reason}); closing the agent connection for {self.conversation_id}")
        except RuntimeError:
            pass

    async def start(self) -> None:
        """Start the FastAPI WebSocket server (non-blocking)."""
        if self._running:
            logger.warning("Server already running")
            return

        await asyncio.to_thread(self._check_agent)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.01)
        logger.info(f"Cartesia managed server started on ws://localhost:{self.port}")

    def _check_agent(self) -> None:
        """Fetch the managed agent and refuse to run if it does not match the run config."""
        agent = cartesia_agents.fetch_agent(self._cartesia_agent_id, self._params)
        cartesia_agents.check_agent_matches_config(
            agent,
            model=self._model,
            language=str(self.language),
            pre_tool_speech=self.pipeline_config.pre_tool_speech,
            turn_end_fallback_time=self._turn_end_fallback_time,
            tool_names=[t["function"]["name"] for t in self.agent.build_tools_for_agent() or []],
        )

    async def _shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None
        # Mix here so the base class skips its slower per-sample mix.
        if self.user_audio_buffer and self.assistant_audio_buffer and not self._audio_buffer:
            self._audio_buffer = bytearray(_mix_pcm16(self.user_audio_buffer, self.assistant_audio_buffer))
        logger.info(f"Cartesia managed server stopped on port {self.port} (call_id={self._call_id})")

    def _session_create(self) -> dict:
        return {
            "type": "session_create",
            # Our input is continuous, so a gap means a stalled EVA process, not silence.
            "audio": {"input_format": f"pcm_{_SAMPLE_RATE}", "continuous_input": True},
            "dynamic_variables": {"system_prompt": self._system_prompt, "initial_message": self.initial_message},
        }

    async def _handle_session(self, websocket: WebSocket) -> None:  # noqa: C901
        """Bridge one simulator session to one agent call."""
        stream_sid = self.conversation_id
        sim_connected = True
        self._loop = asyncio.get_running_loop()
        self._caller_hung_up = asyncio.Event()

        # Simulator turn timestamps (wall-clock ms strings, as the audit log expects).
        user_speech_start_ts: str | None = None
        user_speech_stop_ts: str | None = None
        # The agent's own end of the caller's turn; a latency reference when it precedes user_speech_stop.
        user_turn_end_ms: int | None = None
        pad_user_on_turn_start = True
        record_t0: float | None = None
        last_sim_frame = time.monotonic()
        input_fed_until = last_sim_frame  # monotonic time up to which the agent has input audio

        # Agent output: PCM for the recording and mulaw for the simulator, drained in lockstep.
        pcm_fifo = bytearray()
        mulaw_fifo = bytearray()
        output_ready = asyncio.Event()
        # The resampler holds back a short tail; pace_output flushes it when a burst ends.
        down = soxr.ResampleStream(_SAMPLE_RATE, 8000, 1, dtype="int16", quality="HQ")
        down_pending = False
        up_state = None

        # One segment per spoken agent turn, closed by its turn_ended.
        segment: _SpeechSegment | None = None
        # A model turn starts with the first agent audio after the caller speaks.
        in_model_turn = False
        last_output_audio = 0.0

        def log_pieces(pieces: list[tuple[str, int]]) -> None:
            for text, start_ms in pieces:
                self.audit_log.append_assistant_output(text, timestamp_ms=str(start_ms))

        def flush_segment(was_interrupted: bool = False) -> None:
            nonlocal segment
            if segment is None:
                return
            segment.end_piece()
            pieces = segment.take_pieces()
            if pieces and was_interrupted:
                pieces[-1] = (f"{pieces[-1][0]} [interrupted]", pieces[-1][1])
            log_pieces(pieces)
            if segment.text.strip():
                self._fw_log.llm_response(segment.text.strip(), timestamp_ms=segment.start_ms)
            self._fw_log.turn_end(was_interrupted=was_interrupted)
            segment = None

        def on_agent_audio(pcm: bytes) -> None:
            nonlocal segment, in_model_turn, last_output_audio, down_pending, user_speech_stop_ts, user_turn_end_ms
            if segment is None:
                segment = _SpeechSegment(_now_ms())
            last_output_audio = time.monotonic()
            pcm_fifo.extend(pcm)
            samples = np.frombuffer(pcm, dtype=np.int16)
            mulaw_fifo.extend(audioop.lin2ulaw(down.resample_chunk(samples).tobytes(), 2))
            down_pending = True
            output_ready.set()
            if not in_model_turn:
                in_model_turn = True
                self._fw_log.turn_start(timestamp_ms=segment.start_ms)
                refs = [int(user_speech_stop_ts)] if user_speech_stop_ts else []
                if user_turn_end_ms:
                    refs.append(user_turn_end_ms)
                if refs and self._metrics_log:
                    latency_ms = segment.start_ms - min(refs)
                    if 0 < latency_ms < 30_000:
                        self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)
                user_speech_stop_ts = None
                user_turn_end_ms = None

        def clear_output() -> None:
            nonlocal down_pending
            pcm_fifo.clear()
            mulaw_fifo.clear()
            down.clear()
            down_pending = False

        def flush_resampler_tail() -> None:
            nonlocal down_pending
            tail = down.resample_chunk(np.zeros(0, dtype=np.int16), last=True)
            down.clear()
            down_pending = False
            if len(tail):
                mulaw_fifo.extend(audioop.lin2ulaw(tail.tobytes(), 2))

        url = (
            f"{self._base_url}/v1/agents/websocket/{self._cartesia_agent_id}?cartesia_version={self._cartesia_version}"
        )
        try:
            async with websockets.connect(
                url, additional_headers={"X-API-Key": self._api_key}, max_size=None
            ) as agent_ws:

                async def send_agent(payload: dict) -> None:
                    await agent_ws.send(json.dumps(payload, ensure_ascii=False))

                await send_agent(self._session_create())
                # Audio sent before session_ready is dropped.
                ready = json.loads(await asyncio.wait_for(agent_ws.recv(), timeout=30))
                if ready.get("type") != "session_ready":
                    raise RuntimeError(f"Expected session_ready, got {ready}")
                self._call_id = ready["call_id"]
                logger.info(f"Agent call started: call_id={self._call_id}")

                async def send_input(pcm: bytes) -> None:
                    await send_agent({"type": "audio_input", "audio": base64.b64encode(pcm).decode()})

                async def run_tool(call: dict) -> None:
                    name = call["tool_name"]
                    try:
                        result = await self.execute_tool(name, call["parameters"] or {})
                        if not isinstance(result, str):
                            result = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                        is_error = False
                    except Exception as e:
                        logger.error(f"Tool {name} failed: {e}", exc_info=True)
                        result, is_error = str(e), True
                    if call["expects_response"]:
                        await send_agent(
                            {
                                "type": "client_tool_result",
                                "tool_call_id": call["tool_call_id"],
                                "result": result,
                                "is_error": is_error,
                            }
                        )

                async def forward_user_audio() -> None:
                    nonlocal stream_sid, sim_connected, user_speech_start_ts, user_speech_stop_ts, in_model_turn
                    nonlocal pad_user_on_turn_start, record_t0, last_sim_frame, up_state, input_fed_until
                    try:
                        while sim_connected and self._running:
                            try:
                                raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                            except TimeoutError:
                                continue
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            event = msg.get("event")
                            if event == "start":
                                stream_sid = msg.get("start", {}).get("streamSid", stream_sid)
                            elif event == "stop":
                                sim_connected = False
                                break
                            elif event == "user_speech_start":
                                user_speech_start_ts = msg.get("timestamp_ms")
                                pad_user_on_turn_start = True
                                in_model_turn = False
                            elif event == "user_speech_stop":
                                user_speech_stop_ts = msg.get("timestamp_ms")
                            elif event == "media":
                                mulaw = parse_twilio_media_message(raw)
                                if not mulaw:
                                    continue
                                pcm_8k = audioop.ulaw2lin(mulaw, 2)
                                pcm, up_state = audioop.ratecv(pcm_8k, 2, 1, 8000, _SAMPLE_RATE, up_state)
                                now = time.monotonic()
                                last_sim_frame = now
                                input_fed_until = max(input_fed_until, now) + len(mulaw) / 8000
                                if record_t0 is None:
                                    record_t0 = now
                                if pad_user_on_turn_start:
                                    _pad_buffer_to_walltime(self.user_audio_buffer, now - record_t0, _SAMPLE_RATE)
                                    pad_user_on_turn_start = False
                                self.user_audio_buffer.extend(pcm)
                                await send_input(pcm)
                    except WebSocketDisconnect:
                        logger.info("Simulator WebSocket disconnected")
                    finally:
                        sim_connected = False

                async def feed_input_silence() -> None:
                    nonlocal input_fed_until
                    next_tick = time.monotonic()
                    while sim_connected and self._running:
                        next_tick += CHUNK_DURATION_S
                        await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
                        now = time.monotonic()
                        if now - last_sim_frame > INPUT_SILENCE_GAP_S and input_fed_until < now:
                            await send_input(_SILENCE_FRAME)
                            input_fed_until += CHUNK_DURATION_S
                        elif next_tick < now - 1.0:
                            next_tick = now

                async def receive_agent_events() -> None:
                    nonlocal user_turn_end_ms, segment
                    async for raw in agent_ws:
                        msg = json.loads(raw)
                        event = msg.get("type")
                        if event == "audio_output":
                            on_agent_audio(base64.b64decode(msg["audio"]))
                        elif event == "turn_output_text_delta":
                            if msg["role"] == "assistant":
                                if segment is None:
                                    segment = _SpeechSegment(_now_ms())
                                segment.add_text(msg["text"])
                                log_pieces(segment.take_pieces())
                        elif event == "turn_ended":
                            text = msg["text"].strip()
                            if msg["role"] == "user":
                                user_turn_end_ms = _now_ms()
                                if text:
                                    self.audit_log.append_user_input(
                                        text, timestamp_ms=user_speech_start_ts or str(user_turn_end_ms)
                                    )
                            # The agent also ends a textless turn per tool call; its speech continues through it.
                            elif text or not msg["tool_calls"]:
                                if segment is None and text:
                                    segment = _SpeechSegment(_now_ms())
                                    segment.add_text(msg["text"])
                                flush_segment(was_interrupted=msg["interrupted"])
                        elif event == "client_tool_call":
                            # Words spoken before the call stay before it in the trace.
                            if segment is not None:
                                segment.end_piece()
                                log_pieces(segment.take_pieces())
                            asyncio.create_task(run_tool(msg))
                        elif event == "audio_output_clear":
                            clear_output()
                        elif event == "error":
                            logger.error(f"Agent error: {msg}")
                    logger.info(
                        f"Agent connection closed for {self.conversation_id}: "
                        f"code={agent_ws.close_code} reason={agent_ws.close_reason!r}"
                    )

                async def pace_output() -> None:
                    nonlocal record_t0
                    next_send = time.monotonic()
                    resume = True
                    while self._running:
                        if not mulaw_fifo:
                            if down_pending and time.monotonic() - last_output_audio > 2 * CHUNK_DURATION_S:
                                flush_resampler_tail()
                                if mulaw_fifo:
                                    continue
                            output_ready.clear()
                            try:
                                await asyncio.wait_for(output_ready.wait(), timeout=CHUNK_DURATION_S)
                            except TimeoutError:
                                pass
                            if not mulaw_fifo:
                                resume = True
                            continue
                        now = time.monotonic()
                        if record_t0 is None:
                            record_t0 = now
                        # Place each burst at its real time; append contiguously within it.
                        if resume:
                            _pad_buffer_to_walltime(self.assistant_audio_buffer, now - record_t0, _SAMPLE_RATE)
                            resume = False
                        chunk = bytes(mulaw_fifo[:MULAW_CHUNK_SIZE])
                        del mulaw_fifo[:MULAW_CHUNK_SIZE]
                        self.assistant_audio_buffer.extend(pcm_fifo[:PCM_CHUNK_SIZE])
                        del pcm_fifo[:PCM_CHUNK_SIZE]
                        if sim_connected:
                            try:
                                await websocket.send_text(create_twilio_media_message(stream_sid, chunk))
                            except Exception:
                                return
                        next_send = max(next_send, now) + CHUNK_DURATION_S
                        await asyncio.sleep(max(0.0, next_send - time.monotonic()))

                tasks = {
                    asyncio.create_task(forward_user_audio()): "user_audio",
                    asyncio.create_task(feed_input_silence()): "input_silence",
                    asyncio.create_task(receive_agent_events()): "agent_events",
                    asyncio.create_task(pace_output()): "audio_pacer",
                    asyncio.create_task(self._caller_hung_up.wait()): "caller_hung_up",
                }
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        logger.error(f"Task '{tasks[task]}' failed: {exc}", exc_info=exc)
                    else:
                        logger.info(f"Task '{tasks[task]}' completed")
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                flush_segment()
        except Exception as e:
            logger.error(f"Cartesia managed session error: {e}", exc_info=True)
        finally:
            logger.info(f"Client disconnected from Cartesia managed server (call_id={self._call_id})")
