"""``AssistantRole``: the business-side answering role (one generic class).

A single concrete, provider-agnostic role. It holds a ``Backend`` and works
with *any* backend -- swapping the backend swaps the provider. All provider
specifics (session, audio format, event parsing) live in the backend; this
class owns only the role-common concerns shared by every assistant regardless
of provider:

- the assistant system prompt (built from agent config);
- the agent tool catalog + ``ToolExecutor`` (tool execution stays role-side);
- the ``AuditLog`` and output artifacts (audit_log.json / transcript.jsonl /
  scenario DBs / audio WAVs);
- the counterparty transport: a Twilio-framed WebSocket **server** the user
  simulator connects to, plus real-time output pacing and audio-track
  recording/alignment (all provider-agnostic -- every assistant exposes this
  same Twilio WS).

It consumes only the normalized ``BackendEvent`` stream, so it never touches a
raw provider event.

Plug-in point: mirrors ``AbstractAssistantServer``'s surface (``start`` /
``stop`` / ``get_conversation_stats`` / ``get_final_scenario_db`` /
``notify_conversation_ending``) so the worker swap is 1:1 -- construct
``AssistantRole(backend=factory.create(...), ...)`` instead of
``server_cls(...)`` and keep every downstream call. Wired behind the
``USE_ROLE_BACKEND_OPENAI_REALTIME`` gate in the worker.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.assistant.tools.tool_executor import ToolExecutor, execute_and_log_tool
from eva.backend.base import Backend, BackendEvent, BackendEventType, ToolCallRequest, ToolCallResult
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.role.base import Role
from eva.utils.audio_utils import (
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    pcm16_mix,
    save_audio_track,
    sync_buffer_to_position,
)
from eva.utils.culture import get_initial_message
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# Twilio counterparty-transport constants (provider-agnostic).
MULAW_CHUNK_SIZE = 160  # bytes per chunk (20ms at 8kHz mulaw)
MULAW_CHUNK_DURATION_S = 0.02
# Don't pad the user track to align with the assistant when real user audio
# arrived within this window (the speaking-state flag can go stale under jitter;
# padding then injects a mid-utterance chop). Guard only ever *skips* a pad.
USER_ACTIVE_GUARD_S = 0.3


def _wall_ms() -> str:
    """Current wall-clock time as epoch-milliseconds string."""
    return str(int(round(time.time() * 1000)))


@dataclass
class _UserTurnRecord:
    """State for a single user speech turn (timestamps + transcript flush flag)."""

    speech_started_wall_ms: str = ""
    speech_stopped_wall_ms: str = ""
    transcript: str = ""
    flushed: bool = False


@dataclass
class _AssistantTurnState:
    """Per-response state the role tracks for recording/metrics/logging."""

    first_audio_wall_ms: str | None = None
    audio_was_streamed: bool = False
    responding: bool = False


class AssistantRole(Role):
    """Generic assistant role. Drives any ``Backend``; owns the Twilio WS transport."""

    def __init__(
        self,
        *,
        backend: Backend,
        current_date_time: str,
        pipeline_config: ModelConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
        language: str = "en",
        turn_end_fallback_seconds: float | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.current_date_time = current_date_time
        self.pipeline_config = pipeline_config
        self.agent = agent
        self.agent_config_path = agent_config_path
        self.scenario_db_path = scenario_db_path
        self.output_dir = Path(output_dir)
        self.port = port
        self.conversation_id = conversation_id
        self.language = language
        self.turn_end_fallback_seconds = turn_end_fallback_seconds
        self.initial_message = get_initial_message(language)

        # Core components.
        self.audit_log = AuditLog()  # type: ignore[no-untyped-call]
        self.tool_handler = ToolExecutor(
            tool_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            tool_module_path=self.agent.tool_module_path,
            current_date_time=current_date_time,
        )

        # Recording buffers. Sample rate comes from the backend (provider format).
        self._audio_buffer = bytearray()
        self.user_audio_buffer = bytearray()
        self.assistant_audio_buffer = bytearray()
        self._audio_sample_rate = backend.output_sample_rate

        self._fw_log: FrameworkLogWriter | None = None
        self._metrics_log: MetricsLogWriter | None = None

        # Server state.
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[Any] | None = None
        self._running = False

        # Prompt + generic tool specs (built once); model name for metrics labels.
        self._system_prompt = self.build_prompt()
        self._tool_specs = self._build_tool_specs()
        self._model = (self.pipeline_config.s2s_params or {}).get("model", "")

        # Per-session/turn state.
        self._user_turn: _UserTurnRecord | None = None
        self._assistant_turn = _AssistantTurnState()
        self._stream_sid = ""
        self._user_speaking = False
        self._bot_speaking = False
        self._audio_interface_speech_start_ts: str | None = None
        self._last_user_audio_mono = 0.0

    # ── Prompt / tool specs (role-owned, provider-agnostic) ───────────

    def build_prompt(self) -> str:
        """Build the assistant system prompt from the agent config."""
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt(
            "realtime_agent.system_prompt",
            agent_personality=self.agent.description,
            agent_instructions=self.agent.instructions,
            datetime=self.current_date_time,
        )
        if self.pipeline_config.pre_tool_speech == "auto":
            prompt += "\n\n" + prompt_manager.get_prompt("agent.pre_tool_speech")
        return prompt

    def _build_tool_specs(self) -> list[dict[str, Any]]:
        """Provider-agnostic tool specs from the agent tools (backend formats to its schema)."""
        specs: list[dict[str, Any]] = []
        for tool in self.agent.tools or []:
            specs.append(
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
        return specs

    # ── Role seams ─────────────────────────────────────────────────────

    async def handle_tool_call_request(self, request: ToolCallRequest) -> ToolCallResult:
        """Execute a tool call the backend surfaced and record it in the audit log."""
        result = await execute_and_log_tool(self.tool_handler, self.audit_log, request.name, request.arguments)
        return ToolCallResult(call_id=request.call_id, result=result)

    def record_audio(self, source: str, audio_data: bytes) -> None:
        """Append PCM16 to the named channel buffer (alignment handled by callers)."""
        if source == "user":
            self.user_audio_buffer.extend(audio_data)
        elif source == "assistant":
            self.assistant_audio_buffer.extend(audio_data)

    def notify_conversation_ending(self, reason: str | None = None) -> None:
        """No-op: native-S2S turn-taking needs no early-end signal (see AbstractAssistantServer)."""
        return None

    def get_conversation_stats(self) -> dict[str, Any]:
        return self.audit_log.get_stats()

    def get_initial_scenario_db(self) -> dict[str, Any]:
        return self.tool_handler.original_db

    def get_final_scenario_db(self) -> dict[str, Any]:
        return self.tool_handler.db

    # ── Server lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the FastAPI WebSocket server (non-blocking)."""
        if self._running:
            logger.warning("Assistant role already running")
            return

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

        config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Assistant role started on ws://localhost:{self.port}")

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
        logger.info(f"Assistant role stopped on port {self.port}")

    async def stop(self) -> asyncio.Task[None] | None:
        """Shut down, extract audio, save outputs (mirrors AbstractAssistantServer.stop)."""
        await self._shutdown()
        self._ensure_mixed_audio()

        mixed_audio = bytes(self._audio_buffer)
        user_audio = bytes(self.user_audio_buffer)
        assistant_audio = bytes(self.assistant_audio_buffer)
        sample_rate = self._audio_sample_rate
        self._audio_buffer.clear()
        self.user_audio_buffer.clear()
        self.assistant_audio_buffer.clear()

        self.save_outputs()

        if mixed_audio or user_audio or assistant_audio:
            return asyncio.create_task(
                asyncio.to_thread(self._save_audio_deferred, mixed_audio, user_audio, assistant_audio, sample_rate)
            )
        return None

    # ── Output persistence ────────────────────────────────────────────

    def save_outputs(self) -> None:
        self.audit_log.save(self.output_dir / "audit_log.json")
        self.audit_log.save_transcript_jsonl(self.output_dir / "transcript.jsonl")
        self._save_scenario_dbs()
        logger.info(f"Outputs saved to {self.output_dir}")

    def _ensure_mixed_audio(self) -> None:
        if self._audio_buffer:
            return
        if self.user_audio_buffer and self.assistant_audio_buffer:
            diff_bytes = abs(len(self.user_audio_buffer) - len(self.assistant_audio_buffer))
            diff_ms = diff_bytes / (2 * self._audio_sample_rate) * 1000
            if diff_ms > 500:
                logger.warning(
                    f"Audio buffer length mismatch: user={len(self.user_audio_buffer)} "
                    f"assistant={len(self.assistant_audio_buffer)} diff={diff_ms:.0f}ms — mixed recording may be skewed"
                )
            self._audio_buffer = bytearray(pcm16_mix(bytes(self.user_audio_buffer), bytes(self.assistant_audio_buffer)))
        elif self.user_audio_buffer:
            self._audio_buffer = bytearray(self.user_audio_buffer)
        elif self.assistant_audio_buffer:
            self._audio_buffer = bytearray(self.assistant_audio_buffer)

    def _save_audio_deferred(
        self, mixed_audio: bytes, user_audio: bytes, assistant_audio: bytes, sample_rate: int
    ) -> None:
        save_audio_track(mixed_audio, self.output_dir / "audio_mixed.wav", sample_rate)
        save_audio_track(user_audio, self.output_dir / "audio_user.wav", sample_rate)
        save_audio_track(assistant_audio, self.output_dir / "audio_assistant.wav", sample_rate)
        if mixed_audio or user_audio or assistant_audio:
            logger.info(f"Saved audio files to {self.output_dir} ({len(mixed_audio)} bytes mixed)")

    def _save_scenario_dbs(self) -> None:
        try:
            with open(self.output_dir / "initial_scenario_db.json", "w") as f:
                json.dump(self.get_initial_scenario_db(), f, indent=2, sort_keys=True, default=str, ensure_ascii=False)
            with open(self.output_dir / "final_scenario_db.json", "w") as f:
                json.dump(self.get_final_scenario_db(), f, indent=2, sort_keys=True, default=str, ensure_ascii=False)
            logger.info(f"Saved scenario database states to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error saving scenario database states: {e}", exc_info=True)
            raise

    # ── Session handling (Twilio WS <-> backend) ──────────────────────

    async def _handle_session(self, websocket: WebSocket) -> None:
        logger.info("Client connected to assistant role")
        self._user_turn = None
        self._assistant_turn = _AssistantTurnState()
        self._stream_sid = self.conversation_id
        self._user_speaking = False
        self._bot_speaking = False

        session = None
        try:
            session = await self.backend.open(system_prompt=self._system_prompt, tools=self._tool_specs)
            # Trigger the initial greeting.
            await self.backend.send(session, text=f"Say: '{self.initial_message}'")

            audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()
            forward_task = asyncio.create_task(self._forward_user_audio(websocket, session))
            receive_task = asyncio.create_task(self._process_backend_events(session, audio_output_queue))
            pacer_task = asyncio.create_task(self._pace_audio_output(websocket, audio_output_queue))

            done, pending = await asyncio.wait(
                [forward_task, receive_task, pacer_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            for task in done:
                if task.exception():
                    logger.error(f"Session task failed: {task.exception()}")
        except Exception as e:
            logger.error(f"Assistant session error: {e}", exc_info=True)
        finally:
            if session is not None:
                await self.backend.close(session)
            logger.info("Client disconnected from assistant role")

    async def _pace_audio_output(self, websocket: WebSocket, audio_output_queue: asyncio.Queue[bytes]) -> None:
        """Drain the output queue and forward chunks to Twilio at real-time rate."""
        next_send_time = time.monotonic()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                try:
                    await websocket.send_text(create_twilio_media_message(self._stream_sid, chunk))
                except Exception as e:
                    logger.error(f"Error sending audio to Twilio WS: {e}")
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

    async def _forward_user_audio(self, websocket: WebSocket, session: Any) -> None:
        """Read Twilio media frames and forward audio to the backend."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                event_type = data.get("event")

                if event_type == "start":
                    self._stream_sid = data.get("start", {}).get("streamSid", self.conversation_id)
                    continue
                if event_type == "stop":
                    break
                if event_type == "user_speech_start":
                    self._audio_interface_speech_start_ts = data.get("timestamp_ms")
                    continue
                if event_type != "media":
                    continue

                mulaw_bytes = parse_twilio_media_message(raw)
                if mulaw_bytes is None:
                    continue

                # Twilio 8kHz mulaw -> backend PCM (24kHz converters; the only backend
                # today is 24k — a different-rate backend would need rate-generic utils).
                pcm = mulaw_8k_to_pcm16_24k(mulaw_bytes)
                if not self._bot_speaking:
                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                self.record_audio("user", pcm)
                self._last_user_audio_mono = time.monotonic()

                await self.backend.send(session, audio=pcm)
        except WebSocketDisconnect:
            logger.debug("Twilio WebSocket disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error forwarding user audio: {e}", exc_info=True)

    async def _process_backend_events(self, session: Any, audio_output_queue: asyncio.Queue[bytes]) -> None:
        """Consume normalized backend events and produce audit/transcript/audio + tool results."""
        try:
            async for event in self.backend.receive(session):
                try:
                    await self._handle_backend_event(event, session, audio_output_queue)
                except Exception as e:
                    logger.error(f"Error handling event {event.event_type}: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in backend event loop: {e}", exc_info=True)

    async def _handle_backend_event(
        self, event: BackendEvent, session: Any, audio_output_queue: asyncio.Queue[bytes]
    ) -> None:
        match event.event_type:
            case BackendEventType.INPUT_SPEECH_STARTED:
                self._on_speech_started()
            case BackendEventType.INPUT_SPEECH_STOPPED:
                self._on_speech_stopped()
            case BackendEventType.TRANSCRIPT:
                self._on_transcript(event)
            case BackendEventType.AUDIO_OUTPUT:
                await self._on_audio_output(event.audio or b"", audio_output_queue)
            case BackendEventType.TURN_END:
                self._on_turn_end(event)
            case BackendEventType.TOOL_CALL_REQUEST:
                await self._on_tool_call(event, session)
            case BackendEventType.ERROR:
                logger.error(f"Backend error: {event.error}")
            # OUTPUT_TURN_STARTED / OUTPUT_AUDIO_DONE: not needed by the assistant.

    # ── Event handlers (consume only normalized fields) ───────────────

    def _on_speech_started(self) -> None:
        self._user_speaking = True
        # Start a new user turn only if the previous one was flushed (preserves the
        # original timestamp when VAD fires multiple speech_started per utterance).
        if not self._user_turn or self._user_turn.flushed:
            start_ts = self._audio_interface_speech_start_ts or _wall_ms()
            self._user_turn = _UserTurnRecord(speech_started_wall_ms=start_ts)
            if self._fw_log:
                self._fw_log.turn_start(timestamp_ms=int(start_ts))
            self._audio_interface_speech_start_ts = None

    def _on_speech_stopped(self) -> None:
        self._user_speaking = False
        wall = _wall_ms()
        if self._user_turn:
            self._user_turn.speech_stopped_wall_ms = wall
        else:
            self._user_turn = _UserTurnRecord(speech_stopped_wall_ms=wall)

    def _on_transcript(self, event: BackendEvent) -> None:
        # Assistant records only the inbound (user) transcript; the outbound
        # transcript arrives finalized on TURN_END.
        if event.metadata.get("stream") != "input":
            return
        if event.metadata.get("failed"):
            if self._user_turn and not self._user_turn.flushed:
                ts = self._user_turn.speech_started_wall_ms or None
                self.audit_log.append_user_input("[user speech - transcription unavailable]", timestamp_ms=ts)
                self._user_turn.flushed = True
            return
        transcript = (event.transcript or "").strip()
        if not transcript:
            return
        ts = None
        if self._user_turn:
            ts = self._user_turn.speech_started_wall_ms or None
            self._user_turn.transcript = transcript
            self._user_turn.flushed = True
        self.audit_log.append_user_input(transcript, timestamp_ms=ts)

    async def _on_audio_output(self, pcm16_bytes: bytes, audio_output_queue: asyncio.Queue[bytes]) -> None:
        if not pcm16_bytes:
            return
        if self._assistant_turn.first_audio_wall_ms is None:
            self._assistant_turn.first_audio_wall_ms = _wall_ms()
            self._assistant_turn.responding = True
            self._bot_speaking = True
            # Model response latency: user speech end -> first audio chunk.
            if self._user_turn and self._user_turn.speech_stopped_wall_ms and self._metrics_log:
                latency_ms = int(self._assistant_turn.first_audio_wall_ms) - int(self._user_turn.speech_stopped_wall_ms)
                if 0 < latency_ms < 30_000:
                    self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)

        # Skip the user-track pad while the user track is actively receiving audio.
        user_recently_active = (time.monotonic() - self._last_user_audio_mono) <= USER_ACTIVE_GUARD_S
        if not self._user_speaking and not user_recently_active:
            sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
        self.record_audio("assistant", pcm16_bytes)
        self._assistant_turn.audio_was_streamed = True

        try:
            mulaw_bytes = pcm16_24k_to_mulaw_8k(pcm16_bytes)
            offset = 0
            while offset < len(mulaw_bytes):
                await audio_output_queue.put(mulaw_bytes[offset : offset + MULAW_CHUNK_SIZE])
                offset += MULAW_CHUNK_SIZE
        except Exception as e:
            logger.error(f"Error converting audio for output queue: {e}")

    def _on_turn_end(self, event: BackendEvent) -> None:
        meta = event.metadata
        usage = meta.get("usage")
        if usage and self._metrics_log:
            self._metrics_log.write_token_usage(
                processor="openai_realtime",
                model=self._model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )

        content = (event.transcript or "").strip()

        if meta.get("interrupted"):
            if content:
                text = content + " [interrupted]"
                self.audit_log.append_assistant_output(text, timestamp_ms=self._assistant_turn.first_audio_wall_ms)
                if self._fw_log:
                    self._fw_log.s2s_transcript(text)
                    self._fw_log.turn_end(was_interrupted=True)
            self._reset_assistant_turn()
            return

        if meta.get("cancelled"):
            self._reset_assistant_turn()
            return

        has_fc = bool(meta.get("has_function_calls"))
        audio_was_streamed = self._assistant_turn.audio_was_streamed

        # Skip rules (unchanged from the s2s server): tool-call-only, mixed-no-audio,
        # audio-without-transcript, and empty turns are not logged as assistant output.
        if (not content and has_fc) or (content and not audio_was_streamed and has_fc) or not content:
            self._reset_assistant_turn()
            return

        timestamp = self._assistant_turn.first_audio_wall_ms or _wall_ms()
        self.audit_log.append_assistant_output(content, timestamp_ms=timestamp)
        if self._fw_log:
            self._fw_log.llm_response(content)
            self._fw_log.turn_end(was_interrupted=False)
        self._reset_assistant_turn()

    def _reset_assistant_turn(self) -> None:
        if self._assistant_turn.first_audio_wall_ms is not None:
            self._bot_speaking = False
        self._assistant_turn = _AssistantTurnState()

    async def _on_tool_call(self, event: BackendEvent, session: Any) -> None:
        request = event.tool_call_request
        assert request is not None
        logger.info(f"Tool call: {request.name}({json.dumps(request.arguments, ensure_ascii=False)})")
        result = await self.handle_tool_call_request(request)
        if self._fw_log:
            self._fw_log.write(
                "tool_call",
                {
                    "frame": "tool_call",
                    "tool_name": request.name,
                    "arguments": request.arguments,
                    "result": result.result,
                },
            )
        await self.backend.send(session, tool_result=result)
