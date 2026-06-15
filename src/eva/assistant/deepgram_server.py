"""Deepgram Voice Agent AssistantServer for EVA-Bench.

Bridges between the Twilio-framed WebSocket (user simulator) and Deepgram's
**Voice Agent API** (a unified STT -> LLM -> TTS agent over a single WebSocket)
via the ``deepgram-sdk`` ``client.agent.v1.connect()`` interface.  Audio flows:

    User simulator (8 kHz mulaw)
        -> 24 kHz PCM16 -> Deepgram agent input
    Deepgram agent output (24 kHz PCM16)
        -> 8 kHz mulaw -> User simulator

All tool calls are executed locally via ``ToolExecutor`` (the agent is configured
with *client-side* functions, so Deepgram emits ``FunctionCallRequest`` events and
we reply with ``send_function_call_response``).  ``ConversationText`` events populate
the audit log.

Note: the Voice Agent event stream does not expose token usage, so token usage is
not reported for this framework (latency is still emitted on the first audio chunk
of each turn).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import uvicorn
from deepgram import AsyncDeepgramClient
from deepgram.agent.v1.types.agent_v1send_function_call_response import AgentV1SendFunctionCallResponse
from deepgram.agent.v1.types.agent_v1settings import AgentV1Settings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.models.agents import AgentConfig
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# Deepgram agent runs at 24 kHz PCM16 in both directions (matches the recording rate).
_RECORDING_SAMPLE_RATE = 24000

# Audio output pacing: send 160-byte mulaw chunks (20ms at 8kHz) at real-time rate
# so the user simulator's silence detection works correctly.
MULAW_CHUNK_SIZE = 160  # bytes per chunk (20ms at 8kHz, 1 byte per sample)
MULAW_CHUNK_DURATION_S = 0.02  # 20ms per chunk

# Send a KeepAlive at least this often so Deepgram's ~10s input-audio timeout never
# fires during user silence (e.g. while the agent is speaking).
KEEPALIVE_INTERVAL_S = 5.0

# Defaults for the Voice Agent listen/think/speak providers (overridable via s2s_params).
_DEFAULT_LISTEN_MODEL = "nova-3"
_DEFAULT_THINK_PROVIDER = "open_ai"
_DEFAULT_SPEAK_MODEL = "aura-2-thalia-en"


def _agent_tools_to_deepgram(agent: AgentConfig) -> list[dict[str, Any]] | None:
    """Convert EVA AgentConfig tools to Deepgram ``think.functions`` (client-side).

    Omitting ``endpoint`` marks each function as client-side, so the agent emits a
    ``FunctionCallRequest`` event instead of calling an HTTP endpoint itself.
    """
    if not agent.tools:
        return None

    functions: list[dict[str, Any]] = []
    for tool in agent.tools:
        functions.append(
            {
                "name": tool.function_name,
                "description": f"{tool.name}: {tool.description}",
                "parameters": {
                    "type": "object",
                    "properties": tool.get_parameter_properties(),
                    "required": tool.get_required_param_names(),
                },
            }
        )
    return functions or None


class DeepgramAssistantServer(AbstractAssistantServer):
    """Bridges Twilio WebSocket <-> Deepgram Voice Agent API for EVA-Bench evaluation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Recording sample rate (Deepgram agent runs at 24 kHz)
        self._audio_sample_rate = _RECORDING_SAMPLE_RATE

        s2s_params = self.pipeline_config.s2s_params or {}
        self._api_key: str = s2s_params.get("api_key", "")
        # ``model`` is the exact LLM id sent to Deepgram (required).
        self._think_model: str = s2s_params["model"]
        # Metrics/run_id label, decoupled from the (often long) Deepgram model id:
        # an explicit ``think_label`` if provided, else the model id itself.
        self._model = s2s_params.get("think_label") or self._think_model
        self._think_provider: str = s2s_params.get("think_provider", _DEFAULT_THINK_PROVIDER)
        self._listen_model: str = s2s_params.get("listen_model", _DEFAULT_LISTEN_MODEL)
        self._speak_model: str = s2s_params.get("speak_model", _DEFAULT_SPEAK_MODEL)

        # Build system prompt (same pattern as the other realtime/S2S servers)
        prompt_manager = PromptManager()
        self._system_prompt = prompt_manager.get_prompt(
            "realtime_agent.system_prompt",
            agent_personality=self.agent.description,
            agent_instructions=self.agent.instructions,
            datetime=self.current_date_time,
        )

        self._functions = _agent_tools_to_deepgram(self.agent)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the FastAPI WebSocket server (non-blocking)."""
        if self._running:
            logger.warning("Server already running")
            return

        if not self._api_key:
            raise ValueError("API key required for Deepgram Voice Agent (set s2s_params.api_key)")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Deepgram agent server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the Deepgram agent server."""
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
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._server_task
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        logger.info(f"Deepgram agent server stopped on port {self.port}")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _build_settings(self) -> AgentV1Settings:
        """Build the Voice Agent ``Settings`` message.

        Constructed from a plain dict and validated into the typed model; pydantic
        resolves the discriminated provider unions and produces the correct wire JSON.
        """
        think: dict[str, Any] = {
            "provider": {"type": self._think_provider, "model": self._think_model},
            "prompt": self._system_prompt,
        }
        if self._functions:
            think["functions"] = self._functions

        settings_dict: dict[str, Any] = {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": self._audio_sample_rate},
                "output": {"encoding": "linear16", "sample_rate": self._audio_sample_rate, "container": "none"},
            },
            "agent": {
                "language": self.language,
                "greeting": self.initial_message,
                "listen": {"provider": {"type": "deepgram", "model": self._listen_model}},
                "think": think,
                "speak": {"provider": {"type": "deepgram", "model": self._speak_model}},
            },
        }
        return AgentV1Settings.model_validate(settings_dict)

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Bridge a single Twilio WebSocket session with the Deepgram Voice Agent."""
        logger.info("Client connected to Deepgram agent server")
        # start() always instantiates these before a session can connect; bind to
        # locals so the narrowed (non-None) type is visible inside the nested tasks.
        assert self._fw_log is not None and self._metrics_log is not None
        fw_log = self._fw_log
        metrics_log = self._metrics_log

        stream_sid: str = self.conversation_id
        twilio_connected = True

        # Per-turn assistant text accumulated from ConversationText(role=assistant)
        _assistant_turn_text: list[str] = []

        _in_model_turn = False
        _user_speaking = False
        _user_speech_start_ts: str | None = None  # From the simulator's VAD
        _user_speech_stop_ts: str | None = None  # From the simulator's VAD
        _assistant_turn_start_ts: str | None = None  # Wall-clock ms of first audio chunk

        # Outbound mulaw chunks; drained by the pacer at real-time rate.
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        client = AsyncDeepgramClient(api_key=self._api_key)
        settings = self._build_settings()

        try:
            async with client.agent.v1.connect() as connection:
                logger.info(f"Deepgram agent session connected (think_model={self._think_model})")
                await connection.send_settings(settings)
                fw_log.turn_start()

                # ----- Concurrent tasks -----
                async def _forward_user_audio() -> None:
                    """Read Twilio WS messages, convert audio, send to Deepgram."""
                    nonlocal stream_sid, twilio_connected, _user_speech_start_ts, _user_speech_stop_ts
                    try:
                        while twilio_connected and self._running:
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
                                logger.info(f"Twilio stream started: {stream_sid}")
                            elif event == "stop":
                                logger.info("Twilio stream stopped")
                                twilio_connected = False
                                break
                            elif event == "user_speech_start":
                                _user_speech_start_ts = msg.get("timestamp_ms")
                            elif event == "user_speech_stop":
                                # Record our own wall-clock receipt time rather than the event's
                                # timestamp_ms: the simulator sends user_speech_stop on a monotonic
                                # clock (unlike the wall-clock user_speech_start), so its value can't
                                # be diffed against the wall-clock first-audio time. The event arrives
                                # in ~real time over the local socket, so receipt time is accurate.
                                _user_speech_stop_ts = str(int(time.time() * 1000))
                            elif event == "media":
                                mulaw_bytes = parse_twilio_media_message(raw)
                                if mulaw_bytes is None:
                                    continue
                                pcm_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)
                                if not _in_model_turn:
                                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                                self.user_audio_buffer.extend(pcm_24k)
                                await connection.send_media(pcm_24k)
                    except WebSocketDisconnect:
                        logger.info("Twilio WebSocket disconnected")
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in user audio forwarder: {e}", exc_info=True)
                    finally:
                        twilio_connected = False

                async def _pace_audio_output() -> None:
                    """Drain audio_output_queue and forward chunks at real-time rate."""
                    nonlocal twilio_connected
                    next_send_time = time.monotonic()
                    try:
                        while self._running:
                            try:
                                chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                            except TimeoutError:
                                continue

                            twilio_msg = create_twilio_media_message(stream_sid, chunk)
                            try:
                                await websocket.send_text(twilio_msg)
                            except Exception:
                                twilio_connected = False
                                return

                            now = time.monotonic()
                            if next_send_time <= now:
                                next_send_time = now
                            next_send_time += MULAW_CHUNK_DURATION_S
                            sleep_duration = next_send_time - time.monotonic()
                            if sleep_duration > 0:
                                await asyncio.sleep(sleep_duration)
                    except asyncio.CancelledError:
                        pass

                def _flush_assistant_turn(interrupted: bool) -> None:
                    nonlocal _assistant_turn_text, _in_model_turn, _assistant_turn_start_ts
                    full_text = " ".join(_assistant_turn_text).strip()
                    if full_text:
                        text = f"{full_text} [interrupted]" if interrupted else full_text
                        self.audit_log.append_assistant_output(text, timestamp_ms=_assistant_turn_start_ts)
                        if interrupted:
                            fw_log.s2s_transcript(full_text)
                        else:
                            fw_log.llm_response(full_text)
                    fw_log.turn_end(was_interrupted=interrupted)
                    _in_model_turn = False
                    _assistant_turn_text = []
                    _assistant_turn_start_ts = None

                def _drain_audio_queue() -> None:
                    while not audio_output_queue.empty():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            audio_output_queue.get_nowait()

                async def _process_deepgram_events() -> None:
                    """Consume events from the Deepgram agent session.

                    We iterate the underlying websocket directly and dispatch on the
                    raw ``type`` field instead of the SDK's typed iterator. In
                    deepgram-sdk 6.1.x the agent response-union deserialization is not
                    discriminated by ``type``: it mis-constructs every JSON event as the
                    same model, so isinstance-based dispatch silently drops transcripts
                    and tool-call requests. Parsing the JSON ourselves is deterministic.
                    Binary frames (TTS audio) are delivered as ``bytes`` unchanged.
                    """
                    nonlocal _assistant_turn_text, _in_model_turn, _user_speaking
                    nonlocal _user_speech_start_ts, _user_speech_stop_ts, _assistant_turn_start_ts
                    try:
                        async for raw in connection._websocket:
                            if not self._running:
                                break

                            # --- Raw TTS audio output (24 kHz PCM16) ---
                            if isinstance(raw, bytes):
                                if not raw:
                                    continue
                                if not _in_model_turn:
                                    _in_model_turn = True
                                    _user_speaking = False
                                    _assistant_turn_start_ts = str(int(round(time.time() * 1000)))
                                    fw_log.turn_start()

                                    # model_response latency: user speech end -> first audio.
                                    # Absent on the initial greeting (model-initiated) turn.
                                    if _user_speech_stop_ts:
                                        latency_ms = int(_assistant_turn_start_ts) - int(_user_speech_stop_ts)
                                        if 0 < latency_ms < 30_000:
                                            metrics_log.write_latency("model_response", latency_ms / 1000, self._model)
                                    _user_speech_stop_ts = None

                                if not _user_speaking:
                                    sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                                self.assistant_audio_buffer.extend(raw)

                                if twilio_connected:
                                    try:
                                        mulaw = pcm16_24k_to_mulaw_8k(raw)
                                    except Exception as conv_err:
                                        logger.warning(f"Audio conversion error ({len(raw)} bytes): {conv_err}")
                                        continue
                                    offset = 0
                                    while offset < len(mulaw):
                                        await audio_output_queue.put(mulaw[offset : offset + MULAW_CHUNK_SIZE])
                                        offset += MULAW_CHUNK_SIZE
                                continue

                            # --- JSON control / transcript events ---
                            try:
                                event = json.loads(raw)
                            except (json.JSONDecodeError, TypeError):
                                continue
                            event_type = event.get("type")

                            # Conversation transcripts (final per turn)
                            if event_type == "ConversationText":
                                text = (event.get("content") or "").strip()
                                if not text:
                                    continue
                                if event.get("role") == "user":
                                    _user_speaking = False
                                    logger.info(f"User transcription: {text}")
                                    self.audit_log.append_user_input(text, timestamp_ms=_user_speech_start_ts)
                                    _user_speech_start_ts = None
                                else:
                                    _assistant_turn_text.append(text)

                            # Agent finished speaking -> end of assistant turn
                            elif event_type == "AgentAudioDone":
                                logger.debug("Deepgram agent audio done")
                                _flush_assistant_turn(interrupted=False)

                            # User barge-in
                            elif event_type == "UserStartedSpeaking":
                                if _in_model_turn:
                                    logger.debug("User barge-in during agent turn")
                                    _user_speaking = True
                                    _drain_audio_queue()
                                    _flush_assistant_turn(interrupted=True)

                            # Client-side tool calls
                            elif event_type == "FunctionCallRequest":
                                for fn in event.get("functions", []):
                                    raw_args = fn.get("arguments")
                                    try:
                                        arguments = json.loads(raw_args) if raw_args else {}
                                    except json.JSONDecodeError:
                                        arguments = {}
                                    fn_name = fn.get("name", "")
                                    logger.info(f"Tool call: {fn_name}({json.dumps(arguments)})")
                                    result = await self.execute_tool(fn_name, arguments)
                                    await connection.send_function_call_response(
                                        AgentV1SendFunctionCallResponse(
                                            type="FunctionCallResponse",
                                            id=fn.get("id"),
                                            name=fn_name,
                                            content=json.dumps(result),
                                        )
                                    )

                            elif event_type in ("Error", "FatalError"):
                                logger.error(f"Deepgram agent error: {event.get('description')}")
                            elif event_type == "Warning":
                                logger.warning(f"Deepgram agent warning: {event.get('description')}")

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in Deepgram event processor: {e}", exc_info=True)

                async def _send_keepalives() -> None:
                    """Keep the Deepgram input stream alive during user silence.

                    The user simulator is half-duplex and stops sending mic audio while
                    the agent is speaking. Without input, Deepgram closes the session with
                    a "did not receive audio within our timeout" error (~10s). Periodic
                    KeepAlive messages reset that timer; they are no-ops when audio flows.
                    """
                    try:
                        while self._running and twilio_connected:
                            await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                            try:
                                await connection.send_keep_alive()
                            except Exception:
                                break
                    except asyncio.CancelledError:
                        pass

                user_task = asyncio.create_task(_forward_user_audio())
                events_task = asyncio.create_task(_process_deepgram_events())
                pacer_task = asyncio.create_task(_pace_audio_output())
                keepalive_task = asyncio.create_task(_send_keepalives())

                done, pending = await asyncio.wait(
                    [user_task, events_task, pacer_task, keepalive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                def _task_name(t: asyncio.Task[None]) -> str:
                    if t is user_task:
                        return "user_audio"
                    if t is events_task:
                        return "deepgram_events"
                    if t is keepalive_task:
                        return "keepalive"
                    return "audio_pacer"

                for task in done:
                    exc = task.exception()
                    if exc:
                        logger.error(f"Task '{_task_name(task)}' failed: {exc}", exc_info=exc)
                    else:
                        logger.info(f"Task '{_task_name(task)}' completed normally")

                for task in pending:
                    logger.info(f"Cancelling pending task '{_task_name(task)}'")
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        except Exception as e:
            logger.error(f"Deepgram agent session error: {e}", exc_info=True)
        finally:
            logger.info("Client disconnected from Deepgram agent server")
