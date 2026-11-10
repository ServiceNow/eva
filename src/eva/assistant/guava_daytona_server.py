"""Guava Daytona AssistantServer for EVA-Bench.

Bridges between a Twilio-framed WebSocket (user simulator) and a Guava Daytona
voice bot.  Audio flows:

    User simulator (8 kHz mulaw)
        -> 8 kHz PCM16 -> Guava Daytona bot
    Guava Daytona bot (8 kHz PCM16)
        -> 8 kHz mulaw -> User simulator

Tool calls are executed locally via ToolExecutor and transcript events from the
bot populate the audit log.
"""

from __future__ import annotations

import asyncio
import audioop
import contextlib
import json
import os
import time
from pathlib import Path

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.guava_sip.bridge import GuavaSipBridge
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils.audio_utils import (
    create_twilio_media_message,
    parse_twilio_media_message,
    sync_buffer_to_position,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# 8 kHz end to end (matches EVA's Twilio transport rate).
SAMPLE_RATE = 8000

# Output pacing: 160-byte mulaw chunks (20 ms @ 8 kHz) drained at real time.
MULAW_CHUNK_SIZE = 160
MULAW_CHUNK_DURATION_S = 0.02

# Forward only speech to the caller, so the caller's turn detector can fire.
ASSISTANT_VAD_RMS_THRESHOLD = 200
ASSISTANT_VAD_HANGOVER_S = 0.3

_DEFAULT_VOICE = "grace"
# Seconds to let the bot come up before we dial.
_DEFAULT_LISTEN_WARMUP_S = 5.0


class GuavaDaytonaAssistantServer(AbstractAssistantServer):
    """Bridge a user-simulator WebSocket with a Guava Daytona voice bot."""

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
        self._audio_sample_rate = SAMPLE_RATE

        s2s = self.pipeline_config.s2s_params or {}
        # Each setting comes from s2s_params, falling back to its env var.
        self._endpoint = s2s.get("endpoint") or os.environ.get("GUAVA_DAYTONA_ENDPOINT", "")
        self._api_key = s2s.get("api_key") or os.environ.get("GUAVA_API_KEY", "")
        self._base_url = s2s.get("base_url") or os.environ.get("GUAVA_BASE_URL", "")
        self._sip_host = s2s.get("sip_host") or os.environ.get("GUAVA_SIP_HOST", "")
        self._sip_port = int(s2s.get("sip_port", 5060))
        self._voice = s2s.get("voice", _DEFAULT_VOICE)
        self._listen_warmup_s = float(s2s.get("listen_warmup_s", _DEFAULT_LISTEN_WARMUP_S))
        self._model = s2s.get("model", "guava_daytona")

        missing = [
            f"s2s_params['{key}'] (or ${env})"
            for key, env, value in (
                ("endpoint", "GUAVA_DAYTONA_ENDPOINT", self._endpoint),
                ("api_key", "GUAVA_API_KEY", self._api_key),
                ("base_url", "GUAVA_BASE_URL", self._base_url),
                ("sip_host", "GUAVA_SIP_HOST", self._sip_host),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"GuavaDaytonaAssistantServer is missing: {', '.join(missing)}")

        if self._endpoint.startswith("ws://"):
            logger.warning(
                "API key is being sent over an unencrypted WebSocket (ws://). "
                "Consider using wss:// for remote endpoints to protect your key."
            )

        self._bridge: GuavaSipBridge | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._guava_ws = None  # control channel

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
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
        logger.info("Guava Daytona server started on ws://localhost:%d", self.port)

    async def _shutdown(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._bridge:
            with contextlib.suppress(Exception):
                await self._bridge.close()
        if self._guava_ws is not None:
            with contextlib.suppress(Exception):
                await self._guava_ws.close()

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except (TimeoutError, asyncio.CancelledError):
                    self._server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._server_task
            self._server = None
            self._server_task = None
        logger.info("Guava Daytona server stopped on port %d", self.port)

    # ------------------------------------------------------------------
    # session config payload
    # ------------------------------------------------------------------

    def _build_session_config(self) -> dict:
        tools = []
        for spec in self.agent.build_tools_for_agent() or []:
            fn = spec.get("function", {})
            name = fn.get("name")
            if name:
                tools.append(
                    {
                        "name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )

        agent_name = self.agent.name or "EVA Agent"
        return {
            "event": "session.config",
            "agent_name": agent_name,
            "agent_org": "EVA Bench",
            "agent_purpose": self.agent.description or self.agent.role or "",
            "voice": self._voice,
            "system_prompt": self._build_system_prompt(),
            "task_objective": (
                f"You are {agent_name}, a voice agent helping the caller. "
                "Greet the caller, find out what they need, and help them to completion. "
                "A back-office assistant answers questions and performs lookups on your behalf."
            ),
            "greeting_instruction": (f"Warmly greet the caller as {agent_name} and ask how you can help them today."),
            # Shared token checked by the Guava endpoint; set $GUAVA_APP_TOKEN.
            "auth_token": os.environ.get("GUAVA_APP_TOKEN", ""),
            "api_key": self._api_key,
            "base_url": self._base_url,
            "sip_host": self._sip_host,
            "sip_port": self._sip_port,
            "tools": tools,
        }

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Bridge one user-simulator session with a Guava bot."""
        logger.info("Simulator connected; requesting a bot from %s", self._endpoint)
        self._session_loop = asyncio.get_running_loop()

        stream_sid: str = self.conversation_id
        twilio_connected = True

        # Per-turn state shared across tasks via nonlocal.
        _in_model_turn = False
        _user_speaking = False
        _user_speech_stop_ts: str | None = None  # simulator VAD: user speech stop (wall ms)
        _assistant_turn_start_ts: str | None = None
        # Latency fallback when the sim's user_speech_stop VAD event is absent.
        self._last_user_final_ms = None

        # utterance_ids already seen: repeat IDs are partial transcripts that update
        # the last user turn instead of adding a new one.
        _caller_seen: dict[str, bool] = {}

        # Outbound mulaw chunks; the pacer drains at real time.
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Caller audio waiting to go to Guava; the pump sends it every 20 ms.
        user_pcm_buffer = bytearray()

        # Last monotonic time we heard real energy in the bot's inbound audio.
        _last_assistant_voice_ts = 0.0

        # ── Bring up the bot + dial it ───────────────────────────────────
        try:
            guava_ws = await websockets.connect(self._endpoint)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to connect to control endpoint: %s", e)
            return
        self._guava_ws = guava_ws

        try:
            await guava_ws.send(json.dumps(self._build_session_config()))

            # Wait for the ready message carrying the session code.
            try:
                ready_raw = await asyncio.wait_for(guava_ws.recv(), timeout=60.0)
                ready = json.loads(ready_raw)
            except (TimeoutError, json.JSONDecodeError) as e:
                logger.error("No/invalid ready message from control endpoint: %s", e)
                return
            if ready.get("event") == "error":
                logger.error("Setup error: %s", ready.get("message"))
                return
            if ready.get("event") != "session.ready":
                logger.error("Unexpected first message from control endpoint: %s", ready.get("event"))
                return
            sip_code = ready.get("sip_code", "")
            if not sip_code:
                logger.error("ready message did not carry a session code")
                return

            logger.info("Bot ready (code=%s…); warming up %.1fs", sip_code[:16], self._listen_warmup_s)
            await asyncio.sleep(self._listen_warmup_s)

            self._bridge = GuavaSipBridge(sip_code, sip_host=self._sip_host, sip_port=self._sip_port)
            await self._bridge.connect()
            logger.info("Guava call connected (code=%s…, model=%s)", sip_code[:16], self._model)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to establish Guava call: %s", e, exc_info=True)
            return

        bridge = self._bridge

        # ── Concurrent tasks ─────────────────────────────────────────────

        async def _forward_user_audio() -> None:
            """Twilio WS -> record user track -> send to the Guava bot."""
            nonlocal stream_sid, twilio_connected
            nonlocal _user_speech_stop_ts, _user_speaking, _in_model_turn
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
                        logger.info("Twilio stream started: %s", stream_sid)
                    elif event == "stop":
                        logger.info("Twilio stream stopped")
                        twilio_connected = False
                        break
                    elif event == "user_speech_start":
                        _user_speaking = True
                        _in_model_turn = False
                        self._fw_log.turn_start()
                    elif event == "user_speech_stop":
                        _user_speech_stop_ts = msg.get("timestamp_ms")
                        _user_speaking = False
                    elif event == "media":
                        mulaw_bytes = parse_twilio_media_message(raw)
                        if mulaw_bytes is None:
                            continue
                        # mulaw 8k -> PCM16 (8 kHz)
                        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                        # Record user track (keep the two tracks time-aligned).
                        if not _in_model_turn:
                            sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                        self.user_audio_buffer.extend(pcm_8k)
                        user_pcm_buffer.extend(pcm_8k)
            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected")
                twilio_connected = False
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.error("Error in user audio forwarder: %s", e, exc_info=True)
            finally:
                twilio_connected = False

        async def _process_guava_audio() -> None:
            """Guava inbound audio -> record assistant track -> mulaw out; latency."""
            nonlocal _in_model_turn, _user_speaking, _assistant_turn_start_ts
            nonlocal _user_speech_stop_ts, twilio_connected, _last_assistant_voice_ts
            try:
                while self._running:
                    try:
                        pcm_8k = await asyncio.wait_for(bridge.recv_audio(), timeout=2.0)
                    except TimeoutError:
                        continue
                    if not pcm_8k:
                        continue

                    if not _in_model_turn:
                        _in_model_turn = True
                        _assistant_turn_start_ts = str(int(round(time.time() * 1000)))
                        self._fw_log.turn_start()
                        # Model response latency: user speech end -> first audio.
                        stop_ts = _user_speech_stop_ts or self._last_user_final_ms
                        if stop_ts and self._metrics_log:
                            latency_ms = int(_assistant_turn_start_ts) - int(stop_ts)
                            if 0 < latency_ms < 30_000:
                                self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)
                        _user_speech_stop_ts = None

                    # Record every chunk, including silence, kept aligned with the user track.
                    if not _user_speaking:
                        sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                    self.assistant_audio_buffer.extend(pcm_8k)

                    now = time.monotonic()
                    if audioop.rms(pcm_8k, 2) >= ASSISTANT_VAD_RMS_THRESHOLD:
                        _last_assistant_voice_ts = now
                    forward_to_caller = (now - _last_assistant_voice_ts) <= ASSISTANT_VAD_HANGOVER_S

                    if twilio_connected and forward_to_caller:
                        try:
                            mulaw = audioop.lin2ulaw(pcm_8k, 2)
                        except Exception as conv_err:  # noqa: BLE001
                            logger.warning("Audio conversion error (%d bytes): %s", len(pcm_8k), conv_err)
                            continue
                        offset = 0
                        while offset < len(mulaw):
                            await audio_output_queue.put(mulaw[offset : offset + MULAW_CHUNK_SIZE])
                            offset += MULAW_CHUNK_SIZE
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.error("Error in Guava audio processor: %s", e, exc_info=True)

        async def _pace_audio_output() -> None:
            """Drain audio_output_queue to the simulator at real-time rate."""
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
                    except Exception:  # noqa: BLE001
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

        async def _pump_user_to_guava() -> None:
            """Send one 20 ms frame to Guava every 20 ms (caller audio, or silence)."""
            FRAME_BYTES = 320  # 20 ms @ 8 kHz PCM16
            SILENCE = b"\x00" * FRAME_BYTES
            next_send = time.monotonic()
            try:
                while self._running and twilio_connected:
                    if len(user_pcm_buffer) >= FRAME_BYTES:
                        frame = bytes(user_pcm_buffer[:FRAME_BYTES])
                        del user_pcm_buffer[:FRAME_BYTES]
                    else:
                        frame = SILENCE
                    await bridge.send_audio(frame)
                    next_send += 0.02
                    delay = next_send - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    elif delay < -0.1:  # fell far behind; resync the clock
                        next_send = time.monotonic()
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.error("Error in user->guava pump: %s", e, exc_info=True)

        async def _receive_control() -> None:
            """Control channel -> tool calls / transcripts / lifecycle."""
            nonlocal twilio_connected, _caller_seen
            try:
                async for raw in guava_ws:
                    if not self._running:
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event = msg.get("event")

                    if event == "tool_call":
                        tc_id = msg.get("tool_call_id", "")
                        name = msg.get("name", "")
                        arguments = msg.get("arguments", {})
                        try:
                            result = await self.execute_tool(name, arguments)
                        except Exception as exc:  # noqa: BLE001
                            result = {"error": str(exc)}
                        await guava_ws.send(
                            json.dumps(
                                {
                                    "event": "tool_result",
                                    "tool_call_id": tc_id,
                                    "result": result,
                                }
                            )
                        )

                    elif event == "transcript.user":
                        text = msg.get("text", "")
                        uid = msg.get("utterance_id", "")
                        if text:
                            self._last_user_final_ms = str(int(round(time.time() * 1000)))
                            if uid and uid in _caller_seen:
                                self.audit_log.update_last_user_input(text)
                            else:
                                self.audit_log.append_user_input(text)
                                if uid:
                                    _caller_seen[uid] = True

                    elif event == "transcript.assistant":
                        text = msg.get("text", "")
                        if text:
                            self.audit_log.append_assistant_output(text)
                            self._fw_log.llm_response(text)

                    elif event == "hangup":
                        logger.info("Guava bot hung up")
                        twilio_connected = False
                        break

                    elif event == "error":
                        logger.error("Control error: %s", msg.get("message"))
                        twilio_connected = False
                        break
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.error("_receive_control error: %s", e, exc_info=True)
            finally:
                twilio_connected = False

        user_task = asyncio.create_task(_forward_user_audio())
        guava_task = asyncio.create_task(_process_guava_audio())
        pacer_task = asyncio.create_task(_pace_audio_output())
        pump_task = asyncio.create_task(_pump_user_to_guava())
        control_task = asyncio.create_task(_receive_control())
        all_tasks = [user_task, guava_task, pacer_task, pump_task, control_task]

        try:
            done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

            names = {
                user_task: "user_audio",
                guava_task: "guava_audio",
                pacer_task: "audio_pacer",
                pump_task: "user_pump",
                control_task: "control",
            }
            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    logger.error("Task '%s' failed: %s", names.get(task, "?"), exc, exc_info=exc)
                else:
                    logger.info("Task '%s' completed normally", names.get(task, "?"))

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            with contextlib.suppress(Exception):
                await guava_ws.close()
            logger.info("Simulator disconnected from Guava Daytona server")
