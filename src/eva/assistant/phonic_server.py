"""Phonic speech-to-speech AssistantServer for EVA-Bench.

Bridges between the Twilio-framed WebSocket (user simulator) and Phonic's STS
API (``wss://api.phonic.ai/v1/sts/ws``) via the official ``phonic`` Python SDK.
Audio flows:

    User simulator (8 kHz mulaw)
        -> Phonic input (input_format=mulaw_8000, forwarded as-is)
    Phonic output (24 kHz PCM16, output_format=pcm_24000)
        -> 8 kHz mulaw -> User simulator

Phonic emits native transcripts for both sides: ``input_text`` (user) and the
``text`` field of each ``audio_chunk`` (assistant). Tools are declared inline
as ``custom_websocket`` tools, so Phonic sends ``tool_call`` and we answer with
``tool_call_output`` after executing via the local ToolExecutor.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from phonic import AsyncPhonic, AudioChunkPayload, ConfigPayload, ToolCallOutputPayload

from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils.audio_utils import (
    MulawStreamUpsampler,
    create_twilio_media_message,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Phonic output rate (output_format=pcm_24000); also the recording rate.
_RECORDING_SAMPLE_RATE = 24000

# Audio output pacing: 160-byte mulaw chunks (20 ms at 8 kHz) at real-time rate
# so the user simulator's silence detection works correctly.
MULAW_CHUNK_SIZE = 160
MULAW_CHUNK_DURATION_S = 0.02

# Phonic closes the socket (code 4506) after websocket_timeout_sec without client
# audio, so send mulaw silence (0xFF) whenever user audio stops flowing.
_KEEPALIVE_INTERVAL = 5.0
_MULAW_SILENCE_B64 = base64.b64encode(b"\xff" * MULAW_CHUNK_SIZE).decode("ascii")  # 20 ms


def _now_ms() -> str:
    return str(int(round(time.time() * 1000)))


# ---------------------------------------------------------------------------
# Tool schema helper
# ---------------------------------------------------------------------------


def _agent_tools_to_phonic(agent: AgentConfig, timeout_ms: int) -> list[dict[str, Any]] | None:
    """Convert EVA AgentConfig tools to inline Phonic ``custom_websocket`` tools."""
    if not agent.tools:
        return None

    tools: list[dict[str, Any]] = []
    for tool in agent.tools:
        # Phonic only accepts strict schemas: every property must be listed as
        # required, so optional params are made nullable (nulls dropped on call).
        properties = tool.get_parameter_properties()
        required = set(tool.get_required_param_names())
        for name, prop in properties.items():
            if name not in required:
                properties[name] = {"anyOf": [prop, {"type": "null"}]}
        tools.append(
            {
                "type": "custom_websocket",
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": tool.function_name,
                        "description": f"{tool.name}: {tool.description}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties),
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                "execution_mode": "sync",
                "tool_call_output_timeout_ms": timeout_ms,
            }
        )
    return tools or None


# ---------------------------------------------------------------------------
# Phonic AssistantServer
# ---------------------------------------------------------------------------


class PhonicAssistantServer(AbstractAssistantServer):
    """Bridges the Twilio WebSocket <-> Phonic STS API."""

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

        self._audio_sample_rate = _RECORDING_SAMPLE_RATE

        s2s_params = self.pipeline_config.s2s_params or {}
        self._model = s2s_params.get("model", "phonic-1")
        self._api_key = s2s_params.get("api_key", "")
        if not self._api_key:
            raise ValueError("Missing Phonic API key in s2s_params['api_key']")

        self._voice = s2s_params.get("voice", "sabrina")
        # Agent from scripts/provision_phonic_agents.py; it carries the tools, so
        # they are only declared inline when no agent is given.
        self._agent_name = s2s_params.get("agent")
        self._language_code = (s2s_params.get("language_code") or self.language).split("-")[0]
        self._system_prompt = self._build_system_prompt()
        self._phonic_tools = _agent_tools_to_phonic(agent, int(s2s_params.get("tool_call_output_timeout_ms", 30000)))
        # Remaining s2s_params pass straight through to the Phonic config
        # (e.g. turn_taking, intelligence_level, vad_threshold, audio_speed).
        self._extra_config = {
            k: v
            for k, v in s2s_params.items()
            if k not in ("api_key", "model", "alias", "voice", "agent", "language_code", "tool_call_output_timeout_ms")
        }

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the FastAPI WebSocket server (non-blocking)."""
        if self._running:
            logger.warning("Server already running")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket):
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

        logger.info(f"Phonic server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the Phonic server."""
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

        logger.info(f"Phonic server stopped on port {self.port}")

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    def _build_config(self) -> ConfigPayload:
        config: dict[str, Any] = {
            "model": self._model,
            "voice_id": self._voice,
            "input_format": "mulaw_8000",
            "output_format": "pcm_24000",
            "default_language": self._language_code,
            **self._extra_config,
        }
        if self._agent_name:
            # The provisioned agent's system_prompt / welcome_message are the
            # {{instructions}} / {{initial_message}} template variables
            # (Phonic reserves system_* names for its own system variables).
            config["agent"] = self._agent_name
            config["template_variables"] = {
                "instructions": self._system_prompt,
                "initial_message": self.initial_message,
            }
        else:
            config["system_prompt"] = self._system_prompt
            config["welcome_message"] = self.initial_message
            if self._phonic_tools:
                config["tools"] = self._phonic_tools
        return ConfigPayload(**config)

    async def _handle_session(self, websocket: WebSocket) -> None:  # noqa: C901
        """Bridge a single Twilio WebSocket session with Phonic."""
        logger.info("Client connected to Phonic server")

        stream_sid: str = self.conversation_id
        twilio_connected = True

        # Per-turn state (shared across tasks via nonlocal).
        _in_model_turn = False
        _user_speaking = False
        _user_speech_start_ts: str | None = None  # simulator VAD: user speech start (wall ms)
        _user_speech_stop_ts: str | None = None  # simulator VAD: user speech stop (wall ms)
        # Phonic's own end-of-user-speech; preferred latency reference (see Hydra server).
        _phonic_speech_stop_wall_ms: str | None = None
        _assistant_turn_start_ts: str | None = None
        _asst_text = ""
        _last_audio_sent = time.monotonic()

        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        client = AsyncPhonic(api_key=self._api_key)

        try:
            async with client.conversations.connect() as socket:
                await socket.send_config(self._build_config())
                logger.info(
                    f"Phonic session connected (model={self._model}, voice={self._voice}, "
                    f"tools={len(self._phonic_tools or [])})"
                )

                _user_recording_upsampler = MulawStreamUpsampler(_RECORDING_SAMPLE_RATE)

                async def _forward_user_audio() -> None:
                    """Read Twilio WS messages and forward mulaw audio to Phonic."""
                    nonlocal stream_sid, twilio_connected, _last_audio_sent
                    nonlocal _user_speech_start_ts, _user_speech_stop_ts, _user_speaking
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
                                _user_speaking = True
                                logger.info(f"User speech start: {_user_speech_start_ts}")
                            elif event == "user_speech_stop":
                                _user_speech_stop_ts = msg.get("timestamp_ms")
                                _user_speaking = False
                                logger.info(f"User speech stop: {_user_speech_stop_ts}")
                            elif event == "media":
                                mulaw_bytes = parse_twilio_media_message(raw)
                                if mulaw_bytes is None:
                                    continue

                                if not _in_model_turn:
                                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                                self.user_audio_buffer.extend(_user_recording_upsampler.process(mulaw_bytes))

                                await socket.send_audio_chunk(
                                    AudioChunkPayload(audio=base64.b64encode(mulaw_bytes).decode("ascii"))
                                )
                                _last_audio_sent = time.monotonic()
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

                            try:
                                await websocket.send_text(create_twilio_media_message(stream_sid, chunk))
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

                def _clear_output_queue() -> None:
                    while not audio_output_queue.empty():
                        audio_output_queue.get_nowait()

                async def _process_phonic_events() -> None:  # noqa: C901
                    """Consume events from the Phonic socket."""
                    nonlocal _in_model_turn, _user_speaking
                    nonlocal _user_speech_stop_ts, _phonic_speech_stop_wall_ms
                    nonlocal _assistant_turn_start_ts, _asst_text

                    _forwarded_bytes = 0
                    _dropped_silence_bytes = 0
                    # Phonic streams input_text word by word; collect one user utterance.
                    _user_words: list[str] = []
                    _user_utt_ts: str | None = None

                    def _flush_user() -> None:
                        nonlocal _user_words, _user_utt_ts
                        text = " ".join(_user_words)
                        if text:
                            logger.info(f"User transcript: {text}")
                            self.audit_log.append_user_input(text, timestamp_ms=_user_utt_ts or _now_ms())
                        _user_words = []
                        _user_utt_ts = None

                    def _flush_assistant(text: str, interrupted: bool) -> None:
                        nonlocal _asst_text, _in_model_turn, _assistant_turn_start_ts
                        nonlocal _forwarded_bytes, _dropped_silence_bytes
                        bytes_per_s = 2 * _RECORDING_SAMPLE_RATE
                        logger.info(
                            f"Phonic turn audio: forwarded {_forwarded_bytes / bytes_per_s:.2f}s, "
                            f"dropped {_dropped_silence_bytes / bytes_per_s:.2f}s of silence"
                        )
                        _forwarded_bytes = 0
                        _dropped_silence_bytes = 0
                        text = " ".join(text.split())
                        if text:
                            stamp = _assistant_turn_start_ts or _now_ms()
                            logger.info(f"Assistant transcript: {text}")
                            label = f"{text} [interrupted]" if interrupted else text
                            self.audit_log.append_assistant_output(label, timestamp_ms=stamp)
                            if interrupted:
                                self._fw_log.s2s_transcript(text)
                            else:
                                self._fw_log.llm_response(text)
                        if _in_model_turn:
                            self._fw_log.turn_end(was_interrupted=interrupted)
                        _asst_text = ""
                        _in_model_turn = False
                        _assistant_turn_start_ts = None

                    try:
                        async for msg in socket:
                            if isinstance(msg, bytes):
                                continue
                            etype = getattr(msg, "type", None)
                            if etype != "audio_chunk":
                                logger.info(f"Phonic event: {etype}")

                            if etype == "audio_chunk":
                                if msg.text:
                                    # Speech segments around a tool call arrive unseparated
                                    # ("...for you.Got it"); add the missing space.
                                    if _asst_text[-1:] in (".", "!", "?") and msg.text[0].isupper():
                                        _asst_text += " "
                                    _asst_text += msg.text
                                pcm_24k = base64.b64decode(msg.audio or "")
                                if len(pcm_24k) < 2:
                                    continue
                                # Phonic streams digital silence (exact zeros), ahead of real
                                # time, both between turns and after the speech in a turn ends.
                                # Forwarding it keeps the user simulator from ever detecting the
                                # end of assistant audio, so drop it (sync padding keeps the
                                # recorded tracks aligned).
                                if not any(pcm_24k):
                                    _dropped_silence_bytes += len(pcm_24k)
                                    continue
                                _forwarded_bytes += len(pcm_24k)

                                if not _user_speaking:
                                    sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                                self.assistant_audio_buffer.extend(pcm_24k)

                                if twilio_connected:
                                    mulaw = pcm16_24k_to_mulaw_8k(pcm_24k)
                                    for offset in range(0, len(mulaw), MULAW_CHUNK_SIZE):
                                        await audio_output_queue.put(mulaw[offset : offset + MULAW_CHUNK_SIZE])

                            elif etype == "input_text":
                                text = (msg.text or "").strip()
                                if text:
                                    if not _user_words:
                                        _user_utt_ts = _user_speech_start_ts or _now_ms()
                                    _user_words.append(text)

                            elif etype == "user_started_speaking":
                                _user_speaking = True

                            elif etype == "assistant_started_speaking":
                                _flush_user()
                                # Turns follow Phonic's speaking events, not audio: a stray
                                # non-silent chunk can trail assistant_finished_speaking.
                                if not _in_model_turn:
                                    _in_model_turn = True
                                    _assistant_turn_start_ts = _now_ms()
                                    self._fw_log.turn_start()
                                    # Model response latency: user speech end -> first audio.
                                    _user_end_ref = _phonic_speech_stop_wall_ms or _user_speech_stop_ts
                                    if _user_end_ref and self._metrics_log:
                                        latency_ms = int(_assistant_turn_start_ts) - int(_user_end_ref)
                                        if 0 < latency_ms < 30_000:
                                            self._metrics_log.write_latency(
                                                "model_response", latency_ms / 1000, self._model
                                            )
                                    _user_speech_stop_ts = None
                                    _phonic_speech_stop_wall_ms = None

                            elif etype == "user_finished_speaking":
                                _user_speaking = False
                                _phonic_speech_stop_wall_ms = _now_ms()

                            elif etype == "assistant_finished_speaking":
                                _flush_assistant(_asst_text, interrupted=False)

                            elif etype == "interrupted_response":
                                # Drop audio Phonic generated but the user talked over.
                                _clear_output_queue()
                                _flush_assistant(msg.text or _asst_text, interrupted=True)

                            elif etype == "tool_call":
                                _flush_user()
                                logger.info(
                                    f"Tool call: {msg.tool_name}({json.dumps(msg.parameters, ensure_ascii=False)})"
                                )
                                arguments = {k: v for k, v in (msg.parameters or {}).items() if v is not None}
                                result = await self.execute_tool(msg.tool_name, arguments)
                                await socket.send_tool_call_output(
                                    ToolCallOutputPayload(tool_call_id=msg.tool_call_id, output=result)
                                )

                            elif etype == "tool_call_interrupted":
                                logger.warning(f"Phonic tool call interrupted: {msg.tool_name} ({msg.tool_call_id})")

                            elif etype == "assistant_ended_conversation":
                                logger.info("Phonic assistant ended the conversation")

                            elif etype == "error":
                                logger.error(f"Phonic error: {msg.error} param_errors={msg.param_errors}")

                            elif etype == "warning":
                                logger.warning(f"Phonic warning: {msg.warning}")

                        logger.info("Phonic socket closed")
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in Phonic event processor: {e}", exc_info=True)
                    finally:
                        _flush_user()
                        if _in_model_turn or _asst_text:
                            _flush_assistant(_asst_text, interrupted=True)

                async def _phonic_keepalive() -> None:
                    """Send silence to Phonic when no user audio is flowing."""
                    nonlocal _last_audio_sent
                    try:
                        while self._running:
                            await asyncio.sleep(_KEEPALIVE_INTERVAL)
                            if time.monotonic() - _last_audio_sent >= _KEEPALIVE_INTERVAL:
                                await socket.send_audio_chunk(AudioChunkPayload(audio=_MULAW_SILENCE_B64))
                                _last_audio_sent = time.monotonic()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass  # non-critical; let other tasks drive shutdown

                tasks = {
                    asyncio.create_task(_forward_user_audio()): "user_audio",
                    asyncio.create_task(_process_phonic_events()): "phonic_events",
                    asyncio.create_task(_pace_audio_output()): "audio_pacer",
                    asyncio.create_task(_phonic_keepalive()): "keepalive",
                }

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        logger.error(f"Task '{tasks[task]}' failed: {exc}", exc_info=exc)
                    else:
                        logger.info(f"Task '{tasks[task]}' completed normally")

                for task in pending:
                    logger.info(f"Cancelling pending task '{tasks[task]}'")
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        except Exception as e:
            logger.error(f"Phonic session error: {e}", exc_info=True)
        finally:
            logger.info("Client disconnected from Phonic server")
