"""Apriel Voice Realtime server with native 8-kHz G.711 mu-law transport.

Media is never resampled or re-encoded on the forwarding path. Decoding to
PCM16 is exclusively for the independent, sample-clocked WAV recordings.
"""

import asyncio
import audioop
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.utils.audio_utils import create_twilio_media_message, sync_buffer_to_position
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

SAMPLE_RATE = 8000
PACKET_BYTES = 160  # 20 ms of mu-law; one byte per sample.
PLAYBACK_END_MARK = "eva.playback.complete"


def _wall_ms() -> str:
    return str(round(time.time() * 1000))


@dataclass
class _UserTurn:
    started_ms: str = ""
    stopped_ms: str = ""
    logged: bool = False


@dataclass
class _Response:
    generation: int
    transcript: str = ""
    text_deltas: list[str] = field(default_factory=list)
    first_audio_ms: str = ""
    first_sent_ms: str = ""
    pending_audio: bytearray = field(default_factory=bytearray)
    call_ids: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    done: bool = False
    interrupted: bool = False
    audio_ended: bool = False
    playback_ended: bool = False
    continued: bool = False
    logged: bool = False


class AprielVoiceRealtimeAssistantServer(AbstractAssistantServer):
    """Bridge EVA's telephone media directly to Apriel Voice Realtime."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._params = self.pipeline_config.s2s_params or {}
        self._model = self._params.get("model", "nowllm-speech")
        self._url = self._connection_url()
        self._system_prompt = self._build_system_prompt()
        self._audio_sample_rate = SAMPLE_RATE
        self._stream_sid = self.conversation_id
        self._provider_socket = None
        self._simulator_socket: WebSocket | None = None
        self._session_active = False
        self._session_tasks: set[asyncio.Task] = set()
        self._tool_tasks: set[asyncio.Task] = set()
        self._tool_failed = asyncio.Event()
        self._tool_error: BaseException | None = None
        self._output_queue: asyncio.Queue[tuple[int, str, bytes | None]] = asyncio.Queue()
        self._output_lock = asyncio.Lock()
        self._tool_lock = asyncio.Lock()
        self._generation = 0
        self._responses: dict[str, _Response] = {}
        self._calls: set[str] = set()
        self._users: dict[str, _UserTurn] = {}
        self._current_response_id = ""
        self._current_user_id = ""
        self._caller_started_ms = ""
        self._caller_stopped_ms = ""
        self._input_started_ms: int | None = None
        self._recording_bot_active = False
        self._conversation_ending = False
        self._playback_active = False

    def _connection_url(self) -> str:
        value = self._params.get("url", "ws://127.0.0.1:8000/v1/realtime")
        parsed = urlsplit(value)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc or parsed.username or parsed.fragment:
            raise ValueError("Apriel Voice Realtime requires a ws:// or wss:// URL without credentials or a fragment")
        query = dict(parse_qsl(parsed.query))
        query["model"] = self._model
        return urlunsplit(parsed._replace(query=urlencode(query)))

    def _build_system_prompt(self) -> str:
        manager = PromptManager()
        key = self._params.get("system_prompt_key", "audio_llm_agent.system_prompt")
        if key not in {"agent.system_prompt", "audio_llm_agent.system_prompt", "realtime_agent.system_prompt"}:
            raise ValueError(f"Unsupported Apriel Voice Realtime system_prompt_key: {key!r}")
        prompt = manager.get_prompt(
            key,
            agent_personality=self.agent.description,
            agent_instructions=self.agent.instructions,
            datetime=self.current_date_time,
        )
        if self.pipeline_config.pre_tool_speech == "auto":
            prompt += "\n\n" + manager.get_prompt("agent.pre_tool_speech")
        return prompt

    def _build_session_config(self) -> dict[str, Any]:
        turn_detection = self._params.get("vad_settings", {"type": "semantic_vad"})
        if not isinstance(turn_detection, dict):
            raise ValueError("Apriel Voice Realtime requires backend turn detection; vad_settings must be an object")
        input_transcription = self._params.get("input_transcription", True)
        if not isinstance(input_transcription, bool):
            raise ValueError("Apriel Voice Realtime input_transcription must be a boolean")
        identify_language = self._params.get("transcription_language_identification", False)
        if not isinstance(identify_language, bool):
            raise ValueError("Apriel Voice Realtime transcription_language_identification must be a boolean")
        tools = [
            {
                "type": "function",
                "name": tool.function_name,
                "description": f"{tool.name}: {tool.description}",
                "parameters": {
                    "type": "object",
                    "properties": tool.get_parameter_properties(),
                    "required": tool.get_required_param_names(),
                },
            }
            for tool in self.agent.tools or []
        ]
        return {
            "type": "realtime",
            "instructions": self._system_prompt,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu", "rate": SAMPLE_RATE},
                    "turn_detection": turn_detection,
                    "transcription": (
                        {
                            "model": self._params.get("transcription_model", "gpt-4o-mini-transcribe"),
                            "language_identification": identify_language,
                        }
                        if input_transcription
                        else None
                    ),
                },
                "output": {
                    "format": {"type": "audio/pcmu", "rate": SAMPLE_RATE},
                    "voice": self._params.get("voice", "auto"),
                },
            },
            "tools": tools,
            "tool_choice": "auto",
        }

    def _validate_session(self, session: dict) -> None:
        for direction in ("input", "output"):
            fmt = session.get("audio", {}).get(direction, {}).get("format", {})
            if fmt.get("type") != "audio/pcmu" or fmt.get("rate", SAMPLE_RATE) != SAMPLE_RATE:
                raise ValueError(f"Apriel Voice Realtime did not negotiate 8-kHz mu-law {direction}: {fmt!r}")
        if self._params.get("input_transcription", True) is False:
            if session.get("audio", {}).get("input", {}).get("transcription") is not None:
                raise ValueError("Apriel Voice Realtime did not disable input transcription")
        else:
            transcription = session.get("audio", {}).get("input", {}).get("transcription")
            if not isinstance(transcription, dict) or not transcription.get("model"):
                raise ValueError("Apriel Voice Realtime did not enable input transcription")
            requested_lid = self._params.get("transcription_language_identification", False)
            if transcription.get("language_identification") is not requested_lid:
                raise ValueError(
                    "Apriel Voice Realtime did not confirm transcription language identification; "
                    "update/restart the backend to support this session setting"
                )

    async def start(self) -> None:
        if self._running:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)
        self._app = FastAPI()

        @self._app.websocket("/ws")
        @self._app.websocket("/")
        async def endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        self._server = uvicorn.Server(
            uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning", lifespan="off")
        )
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            if self._server_task.done():
                await self._server_task
                raise RuntimeError("Apriel Voice Realtime local server stopped during startup")
            await asyncio.sleep(0.01)
        logger.info(f"Apriel Voice Realtime server ready on port {self.port}")

    def notify_conversation_ending(self, reason: str | None = None) -> None:
        self._conversation_ending = True

    async def _shutdown(self) -> None:
        self._running = False
        self._conversation_ending = True
        if self._provider_socket is not None:
            await self._provider_socket.close()
        await self._cancel_tasks()
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5)
            except TimeoutError:
                self._server_task.cancel()
                await asyncio.gather(self._server_task, return_exceptions=True)
        self._align_recording_tail()

    async def _cancel_tasks(self) -> None:
        tasks = self._session_tasks | self._tool_tasks
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._session_tasks.clear()
        self._tool_tasks.clear()

    async def _send(self, event: dict) -> None:
        await self._provider_socket.send(json.dumps(event))

    async def _handle_session(self, websocket: WebSocket) -> None:
        if self._session_active:
            await websocket.close(code=1008)
            return
        self._session_active = True
        self._simulator_socket = websocket
        try:
            headers = {}
            if self._params.get("api_key"):
                headers["Authorization"] = f"Bearer {self._params['api_key']}"
            options = self._params.get("websocket_connection_options") or {}
            if not isinstance(options, dict):
                raise ValueError("Apriel Voice Realtime websocket_connection_options must be an object")
            connect_kwargs: dict[str, Any] = dict(options)
            host = urlsplit(self._url).hostname or ""
            if "_" in host:
                # Underscores are illegal in TLS SNI/certificate names, so validate
                # the handshake against the hyphenated name the certificate carries.
                connect_kwargs.setdefault("server_hostname", host.replace("_", "-"))
            async with websockets.connect(
                self._url,
                additional_headers=headers,
                ping_timeout=None,
                max_size=8 * 1024 * 1024,
                **connect_kwargs,
            ) as conn:
                self._provider_socket = conn
                await self._send({"type": "session.update", "session": self._build_session_config()})
                # Never forward media under the server's initial/default format.
                async with asyncio.timeout(self._params.get("session_timeout_seconds", 15)):
                    while True:
                        event = json.loads(await conn.recv())
                        if event.get("type") == "error":
                            raise RuntimeError(f"Apriel Voice Realtime session rejected: {event.get('error')}")
                        if event.get("type") == "session.updated":
                            session = event.get("session", {})
                            self._validate_session(session)
                            self._fw_log.write(
                                "session_config",
                                {
                                    "model": session.get("model", self._model),
                                    "enable_thinking": session.get("enable_thinking"),
                                    "pre_tool_speech": session.get("pre_tool_speech"),
                                    "audio": session.get("audio", {}),
                                },
                            )
                            break
                await self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": f"Say '{self.initial_message}'"}],
                        },
                    }
                )
                await self._send({"type": "response.create"})
                self._session_tasks = {
                    asyncio.create_task(self._forward_user_audio(websocket)),
                    asyncio.create_task(self._receive_events(conn)),
                    asyncio.create_task(self._pace_audio_output(websocket)),
                    asyncio.create_task(self._watch_tool_failures()),
                }
                done, _ = await asyncio.wait(self._session_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
        except (WebSocketDisconnect, websockets.ConnectionClosedOK):
            pass
        except websockets.InvalidStatus as exc:
            logger.error(
                "Apriel Voice Realtime handshake rejected: url=%s status=%s reason=%s headers=%s body=%s",
                self._url,
                exc.response.status_code,
                exc.response.reason_phrase,
                dict(exc.response.headers),
                exc.response.body,
            )
            await websocket.close(code=1011)
        except Exception:
            logger.exception("Apriel Voice Realtime session failed")
            await websocket.close(code=1011)
        finally:
            await self._cancel_tasks()
            self._provider_socket = None
            self._simulator_socket = None
            self._session_active = False
            self._align_recording_tail()
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    async def _forward_user_audio(self, websocket: WebSocket) -> None:
        while True:
            message = json.loads(await websocket.receive_text())
            kind = message.get("event")
            if kind == "stop":
                return
            if kind == "start":
                self._stream_sid = message.get("start", {}).get("streamSid", self.conversation_id)
            elif kind == "user_speech_start":
                self._caller_started_ms = message.get("timestamp_ms", "")
            elif kind == "user_speech_stop":
                self._caller_stopped_ms = message.get("timestamp_ms", "")
            elif kind == "assistant_playback_state":
                await self._report_playback(message.get("active") is True)
            elif kind == "media":
                payload = message.get("media", {}).get("payload", "")
                if not payload:
                    continue
                audio = base64.b64decode(payload, validate=True)
                if self._input_started_ms is None:
                    self._input_started_ms = round(time.time() * 1000)
                # The original base64 payload is forwarded, not decoded/re-encoded.
                await self._send({"type": "input_audio_buffer.append", "audio": payload})
                self.user_audio_buffer.extend(audioop.ulaw2lin(audio, 2))
                if not self._recording_bot_active:
                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))

    async def _report_playback(self, active: bool) -> None:
        if active == self._playback_active:
            return
        self._playback_active = active
        # Optional wire extension; product naming is independent of the protocol.
        event_type = self._params.get("playback_state_event")
        if event_type:
            await self._send({"type": event_type, "active": active})

    async def _receive_events(self, conn) -> None:
        async for raw in conn:
            if not isinstance(raw, str):
                raise ValueError("Apriel Voice Realtime expected base64 JSON audio, not an encoded binary stream")
            await self._handle_event(json.loads(raw))

    def _response(self, event: dict) -> tuple[str, _Response]:
        response_id = event.get("response_id") or event.get("response", {}).get("id") or self._current_response_id
        if not response_id:
            raise ValueError("Apriel Voice Realtime response event has no response ID")
        return response_id, self._responses.setdefault(response_id, _Response(self._generation))

    async def _handle_event(self, event: dict) -> None:
        kind = event.get("type", "")
        if kind == "session.updated":
            self._validate_session(event.get("session", {}))
        elif kind == "error":
            raise RuntimeError(f"Apriel Voice Realtime error: {event.get('error')}")
        elif kind == "response.created":
            self._current_response_id, _ = self._response(event)
        elif kind == "input_audio_buffer.speech_started":
            await self._interrupt_output()
            item_id = event.get("item_id", "")
            self._current_user_id = item_id
            turn = self._users.setdefault(item_id, _UserTurn())
            if not turn.started_ms:
                turn.started_ms = self._caller_started_ms or self._input_timestamp(event.get("audio_start_ms"))
                self._caller_started_ms = ""
                if self._fw_log:
                    self._fw_log.turn_start(timestamp_ms=int(turn.started_ms))
        elif kind == "input_audio_buffer.speech_stopped":
            turn = self._users.setdefault(event.get("item_id", self._current_user_id), _UserTurn())
            turn.stopped_ms = self._caller_stopped_ms or self._input_timestamp(event.get("audio_end_ms"))
            self._caller_stopped_ms = ""
        elif kind in {
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
        }:
            item_id = event.get("item_id", self._current_user_id)
            turn = self._users.setdefault(item_id, _UserTurn())
            text = event.get("transcript", "").strip()
            if kind.endswith(".failed"):
                text = "[user speech - transcription unavailable]"
            if text and not turn.logged:
                self.audit_log.append_user_input(text, timestamp_ms=turn.started_ms or None)
                turn.logged = True
        elif kind == "output_audio_buffer.cleared":
            await self._interrupt_output()
        elif kind in {"response.output_audio.delta", "response.audio.delta"}:
            response_id, response = self._response(event)
            if response.interrupted or response.audio_ended:
                return
            audio = base64.b64decode(event.get("delta", ""), validate=True)
            if not audio:
                return
            if not response.first_audio_ms:
                response.first_audio_ms = _wall_ms()
                user = self._users.get(self._current_user_id)
                if user and user.stopped_ms and self._metrics_log:
                    latency = (int(response.first_audio_ms) - int(user.stopped_ms)) / 1000
                    if latency >= 0:
                        self._metrics_log.write_latency("model_response", latency, self._model)
            response.pending_audio.extend(audio)
            while len(response.pending_audio) >= PACKET_BYTES:
                packet = bytes(response.pending_audio[:PACKET_BYTES])
                del response.pending_audio[:PACKET_BYTES]
                self._output_queue.put_nowait((response.generation, response_id, packet))
        elif kind in {"response.output_audio.done", "response.audio.done"}:
            response_id, response = self._response(event)
            self._finish_audio(response_id, response)
        elif kind in {"response.output_audio_transcript.delta", "response.audio_transcript.delta"}:
            _, response = self._response(event)
            response.text_deltas.append(event.get("delta", ""))
        elif kind in {"response.output_audio_transcript.done", "response.audio_transcript.done"}:
            _, response = self._response(event)
            response.transcript = event.get("transcript", "")
        elif kind == "response.function_call_arguments.done":
            response_id, response = self._response(event)
            self._start_tool(response, event)
        elif kind == "response.done":
            response_id, response = self._response(event)
            if response.done:
                return
            data = event.get("response", {})
            response.done = True
            cancel_current = (
                data.get("status") != "completed"
                and not response.interrupted
                and response.generation == self._generation
            )
            response.interrupted |= data.get("status") != "completed"
            if cancel_current:
                await self._interrupt_output()
            output = data.get("output", [])
            for item in output:
                if item.get("type") == "function_call":
                    self._start_tool(response, item)
            response.transcript = self._response_text(output) or response.transcript
            self._finish_audio(response_id, response)
            if response.playback_ended or response.interrupted:
                self._log_response(response)
            usage = data.get("usage") or {}
            if usage and self._metrics_log:
                self._metrics_log.write_token_usage(
                    processor="apriel_voice_realtime",
                    model=self._model,
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                )
            await self._send_tool_results(response)

    def _input_timestamp(self, offset: int | None) -> str:
        if offset is not None and self._input_started_ms is not None:
            return str(self._input_started_ms + offset)
        return _wall_ms()

    @staticmethod
    def _response_text(output: list[dict]) -> str:
        return "".join(
            part.get("transcript") or part.get("text") or ""
            for item in output
            for part in item.get("content", [])
            if part.get("type") in {"audio", "text", "output_audio", "output_text"}
        ).strip()

    def _log_response(self, response: _Response) -> None:
        if response.logged or not response.first_sent_ms:
            return
        content = response.transcript or "".join(response.text_deltas).strip()
        if not content:
            return
        if response.interrupted:
            content += " [interrupted]"
        self.audit_log.append_assistant_output(content, timestamp_ms=response.first_sent_ms)
        response.logged = True
        if self._fw_log:
            self._fw_log.llm_response(content, timestamp_ms=int(response.first_sent_ms))
            self._fw_log.turn_end(was_interrupted=response.interrupted)

    def _finish_audio(self, response_id: str, response: _Response) -> None:
        if response.audio_ended:
            return
        response.audio_ended = True
        if response.interrupted:
            response.pending_audio.clear()
            return
        if response.pending_audio:
            self._output_queue.put_nowait((response.generation, response_id, bytes(response.pending_audio)))
            response.pending_audio.clear()
        if response.first_audio_ms:
            self._output_queue.put_nowait((response.generation, response_id, None))

    async def _pace_audio_output(self, websocket: WebSocket) -> None:
        deadline = time.monotonic()
        while True:
            generation, response_id, audio = await self._output_queue.get()
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            async with self._output_lock:
                response = self._responses[response_id]
                if generation != self._generation or response.interrupted:
                    continue
                if audio is None:
                    await websocket.send_text(
                        json.dumps(
                            {"event": "mark", "streamSid": self._stream_sid, "mark": {"name": PLAYBACK_END_MARK}}
                        )
                    )
                    self._recording_bot_active = False
                    response.playback_ended = True
                    if response.done:
                        self._log_response(response)
                    continue
                await websocket.send_text(create_twilio_media_message(self._stream_sid, audio))
                if not response.first_sent_ms:
                    response.first_sent_ms = _wall_ms()
                # Record only audio actually forwarded; queued/canceled speech
                # must not appear in the saved assistant track.
                if not self._recording_bot_active:
                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                    self._recording_bot_active = True
                self.assistant_audio_buffer.extend(audioop.ulaw2lin(audio, 2))
                # The caller sends no media (not even silence) while the assistant is
                # speaking, so user_audio_buffer only catches up to real time here.
                sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                deadline = max(deadline, time.monotonic()) + len(audio) / SAMPLE_RATE

    async def _interrupt_output(self) -> None:
        async with self._output_lock:
            self._generation += 1
            for response in self._responses.values():
                if not response.playback_ended or not response.done:
                    response.interrupted = True
                    response.pending_audio.clear()
            while not self._output_queue.empty():
                self._output_queue.get_nowait()
            self._recording_bot_active = False
            if self._simulator_socket is not None:
                await self._simulator_socket.send_text(json.dumps({"event": "clear", "streamSid": self._stream_sid}))

    def _start_tool(self, response: _Response, event: dict) -> None:
        call_id = event.get("call_id", "")
        if not call_id or call_id in self._calls:
            return
        self._calls.add(call_id)
        response.call_ids.append(call_id)
        task = asyncio.create_task(self._run_tool(response, call_id, event))
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_finished)

    def _tool_finished(self, task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            self._tool_error = error
            self._tool_failed.set()

    async def _watch_tool_failures(self) -> None:
        await self._tool_failed.wait()
        raise RuntimeError("Apriel Voice Realtime could not publish a tool result") from self._tool_error

    async def _run_tool(self, response: _Response, call_id: str, event: dict) -> None:
        try:
            arguments = json.loads(event.get("arguments", "{}"))
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            result = await self.execute_tool(event["name"], arguments)
        except Exception as exc:
            logger.exception("Apriel Voice Realtime tool failed")
            result = {"error": str(exc)}
        response.results[call_id] = json.dumps(result, ensure_ascii=False)
        await self._send_tool_results(response)

    async def _send_tool_results(self, response: _Response) -> None:
        async with self._tool_lock:
            if (
                not response.done
                or response.continued
                or not response.call_ids
                or not all(call_id in response.results for call_id in response.call_ids)
            ):
                return
            for call_id in response.call_ids:
                await self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": response.results[call_id],
                        },
                    }
                )
            response.continued = True
            if not response.interrupted and response.generation == self._generation and not self._conversation_ending:
                await self._send({"type": "response.create"})

    def _align_recording_tail(self) -> None:
        size = max(len(self.user_audio_buffer), len(self.assistant_audio_buffer))
        sync_buffer_to_position(self.user_audio_buffer, size)
        sync_buffer_to_position(self.assistant_audio_buffer, size)
