"""Krisp Managed Voice Agent assistant server.

Bridges the user simulator's Twilio-style WebSocket to a published Krisp managed voice agent over the
platform's Twilio-style media stream. The platform runs speech, turn-taking and tool calling; EVA's
tools reach this process as HTTP calls to a small gateway it hosts, routed to the conversation by the
session id header. The stream carries audio only, so after the call the server reads the agent's text
from the platform's session history and places each turn at the audio it was spoken in. Agents are
provisioned by Krisp and checked against the run config before the call.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from eva.assistant.base_server import AbstractAssistantServer, RecordInvalidError
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils import krisp_agents
from eva.utils.audio_utils import create_twilio_media_message, parse_twilio_media_message
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000  # recording rate; the media stream itself is 8 kHz mulaw both ways
MULAW_CHUNK_SIZE = 160  # 20 ms at 8 kHz
CHUNK_DURATION_S = 0.02
TAIL_DRAIN_S = 5.0  # longest agent audio still played to the caller after the platform hangs up
MULAW_SILENCE = b"\xff" * MULAW_CHUNK_SIZE
# The platform streams digital silence between replies; below this RMS a frame is silence.
SILENCE_RMS = 60
# Silent frames inside a reply before it counts as finished (~480 ms).
REPLY_HANGOVER_FRAMES = 12
SPEAKING_GAP_MS = REPLY_HANGOVER_FRAMES * 20 + 100  # a pause shorter than this keeps one reply going
# Queued silent caller frames beyond this are dropped so silence never delays speech.
SILENT_BACKLOG_FRAMES = 3
DEFAULT_GATEWAY_PORT = 8290
CONNECT_TIMEOUT_S = 45.0


class KrispCallInvalidError(RecordInvalidError):
    """Raised from shutdown so outputs are never saved: the worker fails and re-dials the record."""


def _now_ms() -> int:
    return int(round(time.time() * 1000))


def _is_silence(mulaw: bytes) -> bool:
    return audioop.rms(audioop.ulaw2lin(mulaw, 2), 2) < SILENCE_RMS


def _mix_pcm16(user: bytes | bytearray, assistant: bytes | bytearray) -> bytes:
    """Same clipped sum as pcm16_mix, vectorized: its Python loop stalled every live call."""
    a = np.frombuffer(user, dtype="<i2").astype(np.int32)
    b = np.frombuffer(assistant, dtype="<i2").astype(np.int32)
    n = max(len(a), len(b))
    mixed = np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b)))
    return np.clip(mixed, -32768, 32767).astype("<i2").tobytes()


def _bind(port: int) -> socket.socket:
    """Bind here, not in uvicorn: its bind failure calls sys.exit inside the task."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sock.close()
        raise
    return sock


async def _wait_started(server: uvicorn.Server, task: asyncio.Task, what: str) -> None:
    while not server.started:
        if task.done():
            exc = None if task.cancelled() else task.exception()
            raise RuntimeError(f"{what} stopped before it started: {exc!r}")
        await asyncio.sleep(0.01)


def _to_pcm16k(mulaw: bytes, state: Any) -> tuple[bytes, Any]:
    """Decode and upsample one packet, carrying resampler state so packets join without clicks."""
    return audioop.ratecv(audioop.ulaw2lin(mulaw, 2), 2, 1, 8000, _SAMPLE_RATE, state)


def heard_text(message: dict) -> str | None:
    """Agent text as the caller heard it; ``None`` when none of it was heard."""
    text = "".join(c.get("text", "") for c in message.get("content") or [] if isinstance(c, dict)).strip()
    heard = message.get("heard")
    if not isinstance(heard, dict) or not heard.get("interrupted"):
        return text or None
    mark = "" if heard.get("cause") == "tts_error" else " [interrupted]"  # a speech failure is not a barge-in
    chars = heard.get("chars")
    if not isinstance(chars, int) or chars >= len(text):
        return f"{text}{mark}" if text else None
    cut = text[:chars]
    if chars and text[chars : chars + 1].isalnum() and " " in cut:
        cut = cut.rsplit(" ", 1)[0]  # never end on half a word
    cut = cut.strip()
    return f"{cut}{mark}" if cut else None


class _ToolGateway:
    """One HTTP endpoint per process; the platform's tool calls reach the conversation that owns the session."""

    def __init__(self, port: int, secret: str):
        self.port = port
        self.secret = secret
        self._sessions: dict[str, KrispAssistantServer] = {}
        self._server: uvicorn.Server | None = None
        self._socket: socket.socket | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        app = FastAPI()

        @app.get("/healthz")
        async def healthz() -> dict:
            return {"status": "ok"}

        @app.post("/tools/{name}")
        async def tool(name: str, request: Request) -> JSONResponse:
            presented = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(presented.encode(), self.secret.encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            owner = self._sessions.get(request.headers.get(krisp_agents.SESSION_ID_HEADER, ""))
            if owner is None:
                return JSONResponse({"status": "error", "message": "unknown session"}, status_code=404)
            try:
                args = await request.json()
            except ValueError:
                args = {}
            try:
                result = await owner.execute_tool(name, args if isinstance(args, dict) else {})
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}", exc_info=True)
                return JSONResponse({"status": "error", "message": str(e)})
            return JSONResponse(result if isinstance(result, (dict, list)) else {"result": result})

        self._app = app

    async def ensure_started(self) -> None:
        loop = asyncio.get_running_loop()
        if self._server is not None and self.loop is loop:
            return
        self.close()  # a server left on a finished loop accepts connections and never answers
        self._socket = _bind(self.port)
        self.port = self._socket.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(self._app, log_level="warning", lifespan="off"))
        self.loop = loop
        task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        try:
            await _wait_started(self._server, task, "Krisp tool gateway")
        except BaseException:
            self.close()  # a later call must retry, not reuse a server that never answered
            raise
        logger.info(f"Krisp tool gateway listening on :{self.port}")

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._server, self._socket, self.loop = None, None, None

    def bind(self, session_id: str, server: KrispAssistantServer) -> None:
        self._sessions[session_id] = server

    def unbind(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


_gateway: _ToolGateway | None = None
_gateway_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


async def _shared_gateway(params: dict[str, Any]) -> _ToolGateway:
    global _gateway
    async with _gateway_locks.setdefault(asyncio.get_running_loop(), asyncio.Lock()):
        if _gateway is None:
            secret = params.get("gateway_secret") or os.environ.get("KRISP_GATEWAY_SECRET", "")
            if len(secret) < 32:
                raise ValueError(
                    "Set a gateway secret of 32+ chars (s2s_params.gateway_secret or KRISP_GATEWAY_SECRET)"
                )
            _gateway = _ToolGateway(int(params.get("gateway_port", DEFAULT_GATEWAY_PORT)), secret)
        await _gateway.ensure_started()
        return _gateway


class KrispAssistantServer(AbstractAssistantServer):
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
        self._params = self.pipeline_config.s2s_params or {}
        for key in ("api_key", "api_url", "model", "assistant_agent_id"):
            if not self._params.get(key):
                raise ValueError(f"Missing {key} in s2s_params for the Krisp framework")
        self._model = self._params["model"]
        self._agent_id = self._params["assistant_agent_id"]
        krisp_agents.call_date_mode(self._params)
        krisp_agents.preamble_mode(self._params)
        self._session_id: str | None = None
        self._gateway: _ToolGateway | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._caller_hung_up: asyncio.Event | None = None
        self._record_t0: float | None = None
        self._session_start_ms: int | None = None
        # Agent replies as heard, one per burst of agent audio: start/end ms, interrupted.
        self._replies: list[dict[str, Any]] = []
        self._caller_starts: list[int] = []
        self.failure: str | None = None

    @property
    def _call_date(self) -> str:
        return krisp_agents.call_date_mode(self._params)

    def notify_conversation_ending(self, reason: str | None = None) -> None:
        """Hang up when the caller does, so the agent stops talking into EVA's teardown."""
        with contextlib.suppress(RuntimeError):
            if self._loop is not None and self._caller_hung_up is not None:
                self._loop.call_soon_threadsafe(self._caller_hung_up.set)

    def get_conversation_stats(self) -> dict[str, Any]:
        """Caller turns are counted live; the transcript is only written at shutdown."""
        return {**super().get_conversation_stats(), "num_turns": len(self._caller_starts)}

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        if tool_name == krisp_agents.CLOCK_TOOL_NAME:
            return {"status": "success", "current_date_time": self.current_date_time}
        return await super().execute_tool(tool_name, arguments)

    async def start(self) -> None:
        if self._running:
            return
        await asyncio.to_thread(self._check_agent)
        self._gateway = await _shared_gateway(self._params)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        sock = _bind(self.port)
        self._server = uvicorn.Server(uvicorn.Config(self._app, log_level="warning", lifespan="off"))
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve(sockets=[sock]))
        try:
            await _wait_started(self._server, self._server_task, f"Krisp server on port {self.port}")
        except BaseException:
            sock.close()
            raise
        logger.info(f"Krisp server started on ws://localhost:{self.port}")

    def _check_agent(self) -> None:
        """Refuse to run against an agent whose published config differs from the run config."""
        agent = krisp_agents.fetch_agent(self._agent_id, self._params)
        krisp_agents.check_agent_matches_config(
            agent,
            model=self._model,
            language=str(self.language),
            system_prompt=krisp_agents.system_prompt(self.agent, self.pipeline_config.pre_tool_speech, self._call_date),
            initial_message=self.initial_message,
            tool_names=[t["function"]["name"] for t in krisp_agents.eva_tools(self.agent, self._call_date)],
            preamble=krisp_agents.preamble_mode(self._params),
        )

    def _pad(self, buffer: bytearray) -> None:
        """Pad a track to wall time so both tracks share one clock."""
        now = time.monotonic()
        if self._record_t0 is None:
            self._record_t0 = now
        target = int((now - self._record_t0) * _SAMPLE_RATE) * 2
        if len(buffer) < target:
            buffer.extend(b"\x00" * (target - len(buffer)))

    def _record(self, buffer: bytearray, pcm: bytes, at_start: bool) -> None:
        """Align to wall time only where a turn or burst starts; a stall must not split speech."""
        if at_start:
            self._pad(buffer)
        buffer.extend(pcm)

    async def _handle_session(self, websocket: WebSocket) -> None:  # noqa: C901
        """Bridge one simulator session to one platform call."""
        self._loop = asyncio.get_running_loop()
        self._caller_hung_up = asyncio.Event()
        stream_sid = f"MZ{uuid.uuid5(uuid.NAMESPACE_URL, self.conversation_id).hex}"
        sim_connected = True
        to_agent: asyncio.Queue[bytes] = asyncio.Queue()
        to_caller: asyncio.Queue[bytes] = asyncio.Queue()
        speaking = False
        silent_frames = 0
        user_speech_stop_ms: int | None = None

        try:
            call_datetime = self.current_date_time if self._call_date == "session" else None
            session = await asyncio.to_thread(
                krisp_agents.mint_voice_session, self._agent_id, self._params, call_datetime
            )
        except Exception as e:
            self.failure = f"could not open a session: {e}"
            logger.error(self.failure)
            await websocket.close(code=1011)
            return
        self._session_id = session["session_id"]
        assert self._gateway is not None
        self._gateway.bind(self._session_id, self)

        start = {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": stream_sid,
            "start": {
                "streamSid": stream_sid,
                "callSid": f"CA{uuid.uuid4().hex}",
                "accountSid": "ACeva-bench",
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": {"token": session["session_token"]},
            },
        }

        def end_reply(interrupted: bool) -> None:
            nonlocal speaking, silent_frames
            if speaking and self._replies:
                self._replies[-1].update(end_ms=_now_ms(), interrupted=interrupted)
            speaking, silent_frames = False, 0

        try:
            async with websockets.connect(
                session["media_stream_url"],
                max_size=None,
                open_timeout=CONNECT_TIMEOUT_S,
                ping_timeout=180,
                additional_headers={"User-Agent": krisp_agents.USER_AGENT},
            ) as agent_ws:
                await agent_ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
                await agent_ws.send(json.dumps(start))
                self._session_start_ms = _now_ms()
                logger.info(f"Krisp call open: session {self._session_id}")

                pad_user = True
                user_rs: Any = None
                agent_rs: Any = None

                async def forward_user_audio() -> None:
                    nonlocal sim_connected, user_speech_stop_ms, pad_user, user_rs
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
                            if event == "stop":
                                break
                            if event == "user_speech_start":
                                ts = msg.get("timestamp_ms")
                                self._caller_starts.append(int(ts) if str(ts).isdigit() else _now_ms())
                                pad_user = True
                            elif event == "user_speech_stop":
                                ts = msg.get("timestamp_ms")
                                user_speech_stop_ms = int(ts) if str(ts).isdigit() else None
                            elif event == "media":
                                mulaw = parse_twilio_media_message(raw)
                                if not mulaw:
                                    continue
                                silent = _is_silence(mulaw)
                                pcm, user_rs = _to_pcm16k(mulaw, user_rs)
                                self._record(self.user_audio_buffer, pcm, pad_user)
                                pad_user = False
                                for i in range(0, len(mulaw), MULAW_CHUNK_SIZE):
                                    if silent and to_agent.qsize() >= SILENT_BACKLOG_FRAMES:
                                        continue
                                    to_agent.put_nowait(mulaw[i : i + MULAW_CHUNK_SIZE])
                    except WebSocketDisconnect:
                        logger.info("Simulator WebSocket disconnected")
                    finally:
                        sim_connected = False

                async def pace_to_agent() -> None:
                    # One frame every 20 ms for the whole call, silence when the caller has none.
                    next_send = time.monotonic()
                    while True:
                        try:
                            chunk = to_agent.get_nowait()
                        except asyncio.QueueEmpty:
                            chunk = MULAW_SILENCE
                        await agent_ws.send(create_twilio_media_message(stream_sid, chunk))
                        next_send = max(next_send + CHUNK_DURATION_S, time.monotonic() - 0.1)
                        await asyncio.sleep(max(0.0, next_send - time.monotonic()))

                async def receive_agent_audio() -> None:
                    nonlocal speaking, silent_frames, user_speech_stop_ms
                    async for raw in agent_ws:
                        if isinstance(raw, bytes):
                            continue
                        msg = json.loads(raw)
                        event = msg.get("event")
                        if event == "clear":
                            while not to_caller.empty():
                                to_caller.get_nowait()
                            end_reply(interrupted=True)
                        elif event == "media":
                            payload = (msg.get("media") or {}).get("payload")
                            if not payload:
                                continue
                            mulaw = base64.b64decode(payload)
                            if _is_silence(mulaw):
                                if not speaking:
                                    continue
                                silent_frames += max(1, len(mulaw) // MULAW_CHUNK_SIZE)
                                if silent_frames > REPLY_HANGOVER_FRAMES:
                                    end_reply(interrupted=False)
                                    continue
                            else:
                                silent_frames = 0
                                if not speaking:
                                    speaking = True
                                    now = _now_ms()
                                    self._replies.append({"start_ms": now, "end_ms": now, "interrupted": False})
                                    if user_speech_stop_ms and self._metrics_log:
                                        latency_ms = now - user_speech_stop_ms
                                        if 0 < latency_ms < 30_000:
                                            self._metrics_log.write_latency(
                                                "model_response", latency_ms / 1000, self._model
                                            )
                                    user_speech_stop_ms = None
                            for i in range(0, len(mulaw), MULAW_CHUNK_SIZE):
                                to_caller.put_nowait(mulaw[i : i + MULAW_CHUNK_SIZE])
                    logger.info(
                        f"Platform closed the call: code={agent_ws.close_code} reason={agent_ws.close_reason!r}"
                    )

                async def pace_to_caller() -> None:
                    nonlocal agent_rs
                    next_send = time.monotonic()
                    while True:
                        burst_start = to_caller.empty()
                        chunk = await to_caller.get()
                        if not sim_connected:
                            return
                        try:
                            await websocket.send_text(create_twilio_media_message(stream_sid, chunk))
                        except Exception:
                            return
                        # Record what the caller heard, not what was cleared.
                        pcm, agent_rs = _to_pcm16k(chunk, agent_rs)
                        self._record(self.assistant_audio_buffer, pcm, burst_start)
                        next_send = max(next_send + CHUNK_DURATION_S, time.monotonic() - 0.1)
                        await asyncio.sleep(max(0.0, next_send - time.monotonic()))

                tasks = {
                    asyncio.create_task(forward_user_audio()): "user_audio",
                    asyncio.create_task(pace_to_agent()): "agent_pacer",
                    asyncio.create_task(receive_agent_audio()): "agent_audio",
                    asyncio.create_task(pace_to_caller()): "caller_pacer",
                    asyncio.create_task(self._caller_hung_up.wait()): "caller_hung_up",
                }
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        logger.error(f"Task '{tasks[task]}' failed: {exc}", exc_info=exc)
                caller_pacer = next(t for t, name in tasks.items() if name == "caller_pacer")
                if {tasks[t] for t in done} == {"agent_audio"} and not caller_pacer.done():
                    drain_until = time.monotonic() + TAIL_DRAIN_S  # the platform hung up: finish its goodbye
                    while not to_caller.empty() and not caller_pacer.done() and time.monotonic() < drain_until:
                        await asyncio.sleep(CHUNK_DURATION_S)
                    await asyncio.sleep(CHUNK_DURATION_S)
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                end_reply(interrupted=False)
                with contextlib.suppress(Exception):
                    await agent_ws.send(json.dumps({"event": "stop", "streamSid": stream_sid}))
        except Exception as e:
            self.failure = f"media stream error: {e}"
            logger.error(f"Krisp session error: {e}", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()  # a clean close, so the simulator never logs a reset
            logger.info(f"Simulator session ended (session {self._session_id})")

    async def _shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._record_t0 is not None:
            self._pad(self.user_audio_buffer)
            self._pad(self.assistant_audio_buffer)
        try:
            if self._session_id is None and self.failure:
                raise KrispCallInvalidError(self.failure)  # the call never opened, so there is nothing to score
            if self._session_id is not None:
                try:  # stay bound while history flushes: late tool calls must still execute
                    history = await asyncio.to_thread(
                        krisp_agents.fetch_session_history, self._session_id, self._params
                    )
                finally:
                    if self._gateway is not None:
                        self._gateway.unbind(self._session_id)
                self._write_transcript(history)
                if unreached := krisp_agents.unreached_tool_calls(history):
                    self.failure = f"tool gateway unreachable: {'; '.join(unreached)}"
                    raise KrispCallInvalidError(self.failure)  # an infrastructure failure, re-dialled, never scored
        except Exception as e:
            self.failure = self.failure or f"transcript not written: {e}"
            raise KrispCallInvalidError(self.failure) from e  # a record without its transcript must never score
        finally:
            if self._server:
                self._server.should_exit = True
                if self._server_task:
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(self._server_task, timeout=5.0)
                self._server = None
                self._server_task = None
        if self.user_audio_buffer and self.assistant_audio_buffer and not self._audio_buffer:
            self._audio_buffer = bytearray(_mix_pcm16(self.user_audio_buffer, self.assistant_audio_buffer))
        logger.info(f"Krisp server stopped on port {self.port} (session {self._session_id})")

    def _tool_times(self, message_type: str = "tool_call") -> list[int]:
        times = []
        for entry in self.audit_log.transcript:
            if entry.get("message_type") == message_type:
                with contextlib.suppress(TypeError, ValueError):
                    times.append(int(entry.get("timestamp")))
        return sorted(times)

    def _write_transcript(self, history: list[dict]) -> None:
        """Place each history turn on the call's clock: caller words first, each text in the audio it was said in."""
        if not history:
            logger.warning(f"No session history for {self._session_id}; the transcript is empty")
        tools = self._tool_times()
        results = self._tool_times("tool_response")
        replies = sorted(self._replies, key=lambda r: r["start_ms"])
        texts: dict[int, list[str]] = {}
        anchor = self._session_start_ms or 0

        def place(earliest: int, latest: int, text: str) -> int | None:
            """Prefer unclaimed audio starting in (earliest, latest], then audio still playing, then claimed audio."""
            in_range = [i for i, r in enumerate(replies) if earliest < r["start_ms"] <= latest]
            playing = [i for i, r in enumerate(replies) if r["start_ms"] <= earliest <= r["end_ms"] + SPEAKING_GAP_MS]
            for i in [*(i for i in in_range if i not in texts), *playing, *in_range]:
                texts.setdefault(i, []).append(text)
                return replies[i]["start_ms"] if i in in_range else earliest + 1
            return None

        for entry in history:
            ts = int(entry.get("timestamp") or _now_ms())
            turn_tools = [t for t in tools if anchor < t <= ts]
            fillers = [
                (int(f.get("timestamp_ms") or 0), (f.get("text") or "").strip())
                for f in entry.get("fillers") or []
                if (f.get("text") or "").strip()
            ]
            user = entry.get("user_message") or {}
            user_text = "".join(c.get("text", "") for c in user.get("content") or [] if isinstance(c, dict)).strip()
            if user_text:
                spoke = next((s for s in self._caller_starts if anchor < s <= ts), None)
                # No new speech start: these words continue the caller's previous utterance.
                if spoke is None and any(s <= anchor for s in self._caller_starts):
                    spoke = anchor + 1
                bounds = [spoke, turn_tools[0] - 1 if turn_tools else None, ts - 1]
                bounds += [f_ts - 1 for f_ts, _ in fillers if f_ts]
                user_ts = (
                    min(x for x in bounds if x and x > anchor) if any(x and x > anchor for x in bounds) else ts - 1
                )
                self.audit_log.append_user_input(user_text, timestamp_ms=str(user_ts))
                anchor = user_ts
            # Lead-ins play while tools run; the reply is spoken after the turn's last tool.
            for f_ts, text in fillers:
                # Audio may start after a fast tool; the log entry stays before it.
                at = place(anchor, ts, text)
                filler_ts = at if at is not None else max(f_ts, anchor + 1)
                if turn_tools:
                    filler_ts = max(anchor + 1, min(filler_ts, turn_tools[0] - 1))
                self.audit_log.append_assistant_output(text, timestamp_ms=str(filler_ts))
                anchor = filler_ts
            agent_text = heard_text(entry.get("agent_message") or {}) if entry.get("agent_message") else None
            if not agent_text:
                continue
            # The reply speaks the tools' results, so it is logged after the last of them.
            turn_results = [t for t in results if anchor < t <= ts]
            after = max([anchor, *turn_tools[-1:], *turn_results[-1:]])
            # A reply cannot start once the caller has begun the next turn.
            next_caller = min((c for c in self._caller_starts if c > ts), default=ts + 3001)
            at = place(after, min(ts + 3000, next_caller - 1), agent_text)
            reply_ts = at if at is not None else max(ts, after + 1)
            self.audit_log.append_assistant_output(agent_text, timestamp_ms=str(reply_ts))
            anchor = reply_ts
        for i, reply in enumerate(replies):
            self._fw_log.turn_start(timestamp_ms=reply["start_ms"])
            if i in texts:
                self._fw_log.llm_response(" ".join(texts[i]), timestamp_ms=reply["start_ms"])
            self._fw_log.turn_end(was_interrupted=reply["interrupted"], timestamp_ms=reply["end_ms"])

    async def save_outputs(self) -> None:
        await super().save_outputs()
        meta = {
            "krisp_agent_id": self._agent_id,
            "krisp_session_id": self._session_id,
            "bridge_failure": self.failure,
            "agent_replies": len(self._replies),
        }
        (self.output_dir / "krisp_session.json").write_text(json.dumps(meta, indent=2))
