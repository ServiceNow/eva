"""OpenAI Realtime API assistant server implementation.

Uses the OpenAI Python SDK's Realtime API (client.beta.realtime.connect())
to bridge audio between a Twilio-framed WebSocket (user simulator) and the
OpenAI Realtime model.  Handles tool calls via the local ToolExecutor and
records all conversation events in the audit log.
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    pcm16_mix,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# OpenAI Realtime operates at 24 kHz 16-bit mono PCM
OPENAI_SAMPLE_RATE = 24000


def _wall_ms() -> str:
    """Return current wall-clock time as epoch-milliseconds string."""
    return str(int(round(time.time() * 1000)))


@dataclass
class _UserTurnRecord:
    """Tracks state for a single user speech turn."""

    speech_started_wall_ms: str = ""
    speech_stopped_wall_ms: str = ""
    transcript: str = ""
    flushed: bool = False


@dataclass
class _AssistantResponseState:
    """Accumulates state for the current assistant response."""

    transcript_parts: list[str] = field(default_factory=list)
    transcript_done_text: str = ""  # Final text from response.audio_transcript.done
    first_audio_wall_ms: Optional[str] = None
    responding: bool = False
    has_function_calls: bool = False


class OpenAIRealtimeAssistantServer(AbstractAssistantServer):
    """Assistant server backed by the OpenAI Realtime API.

    Exposes a local WebSocket at ``ws://localhost:{port}/ws`` using the Twilio
    frame format so the user simulator can connect as if talking to Twilio.
    Internally bridges audio between Twilio (8 kHz mulaw) and OpenAI Realtime
    (24 kHz PCM16 base64).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._audio_sample_rate = OPENAI_SAMPLE_RATE

        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        self._running: bool = False

        self._fw_log: Optional[FrameworkLogWriter] = None
        self._metrics_log: Optional[MetricsLogWriter] = None

        prompt_manager = PromptManager()
        self._system_prompt: str = prompt_manager.get_prompt(
            "realtime_agent.system_prompt",
            agent_personality=self.agent.description,
            agent_instructions=self.agent.instructions,
            datetime=self.current_date_time,
        )

        self._realtime_tools: list[dict] = self._build_realtime_tools()

        self._user_turn: Optional[_UserTurnRecord] = None
        self._assistant_state = _AssistantResponseState()
        self._stream_sid: str = ""

        # Wall-clock tracking for audio alignment
        self._session_start_wall: float = 0.0  # time.time() when session starts
        self._assistant_audio_last_wall: float = 0.0  # time.time() of last assistant audio write

        self._model: str = self.pipeline_config.s2s

    async def start(self) -> None:
        """Start the FastAPI WebSocket server."""
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

        logger.info(f"OpenAI Realtime server started on ws://localhost:{self.port}")

    async def stop(self) -> None:
        """Stop the server and save all outputs."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        await self.save_outputs()
        logger.info(f"OpenAI Realtime server stopped on port {self.port}")

    async def save_outputs(self) -> None:
        """Save all outputs including mixed audio."""
        user_duration = len(self.user_audio_buffer) / (OPENAI_SAMPLE_RATE * 2) if self.user_audio_buffer else 0.0
        asst_duration = (
            len(self.assistant_audio_buffer) / (OPENAI_SAMPLE_RATE * 2) if self.assistant_audio_buffer else 0.0
        )
        logger.info(
            f"[SILENCE DEBUG] Final buffers: user={user_duration:.2f}s, "
            f"assistant={asst_duration:.2f}s, diff={user_duration - asst_duration:.2f}s"
        )
        # Compute mixed audio from user + assistant tracks
        if self.user_audio_buffer and self.assistant_audio_buffer:
            self._audio_buffer = bytearray(pcm16_mix(bytes(self.user_audio_buffer), bytes(self.assistant_audio_buffer)))

        await super().save_outputs()

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Handle a single WebSocket session.

        1. Accept Twilio WS connection
        2. Connect to OpenAI Realtime API
        3. Configure session (instructions, tools, voice, VAD)
        4. Run two concurrent tasks:
           a. Forward user audio: Twilio WS -> decode mulaw -> PCM16 24kHz base64 -> OpenAI
           b. Process OpenAI events: async for event in conn -> handle each type
        5. On tool call: execute via self.tool_handler, send result back
        6. On audio: decode base64 PCM16 -> record -> encode mulaw -> send to Twilio WS
        """
        logger.info("Client connected to OpenAI Realtime server")

        # Reset per-session state
        self._user_turn = None
        self._assistant_state = _AssistantResponseState()
        self._stream_sid = self.conversation_id
        self._session_start_wall = time.time()
        self._assistant_audio_last_wall = 0.0

        api_key = self.pipeline_config.s2s_params.get("api_key")
        if not api_key:
            raise ValueError("API key required for openai realtime")
        client = AsyncOpenAI(api_key=api_key)

        try:
            async with client.beta.realtime.connect(model=self._model) as conn:
                # Configure the session
                await conn.session.update(
                    session={
                        "modalities": ["text", "audio"],
                        "instructions": self._system_prompt,
                        "voice": self.pipeline_config.s2s_params.get("voice", "marin"),
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": self.pipeline_config.s2s_params.get("transcription_model", "whisper-1"),
                        },
                        # TODO: Add support for client_vad and configurable turn detection params
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 200,
                        },
                        "tools": self._realtime_tools,
                    }
                )

                # Trigger the initial greeting
                await conn.response.create()

                # Run forwarding tasks concurrently
                forward_task = asyncio.create_task(self._forward_user_audio(websocket, conn))
                receive_task = asyncio.create_task(self._process_openai_events(conn, websocket))

                done, pending = await asyncio.wait(
                    [forward_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check for exceptions in completed tasks
                for task in done:
                    if task.exception():
                        logger.error(f"Session task failed: {task.exception()}")

        except Exception as e:
            logger.error(f"OpenAI Realtime session error: {e}", exc_info=True)
        finally:
            logger.info("Client disconnected from OpenAI Realtime server")

    # ── User audio forwarding (Twilio WS -> OpenAI) ──────────────────

    async def _forward_user_audio(self, websocket: WebSocket, conn: Any) -> None:
        """Read Twilio media frames and forward audio to OpenAI Realtime."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                event_type = data.get("event")

                if event_type == "start":
                    # Twilio stream start - extract streamSid
                    self._stream_sid = data.get("start", {}).get("streamSid", self.conversation_id)
                    logger.debug(f"Twilio stream started: streamSid={self._stream_sid}")
                    continue

                if event_type == "stop":
                    logger.debug("Twilio stream stopped")
                    break

                if event_type != "media":
                    continue

                # Extract raw mulaw audio bytes
                mulaw_bytes = parse_twilio_media_message(raw)
                if mulaw_bytes is None:
                    continue

                # Convert 8kHz mulaw -> 24kHz PCM16
                pcm16_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)

                # Record user audio
                self.user_audio_buffer.extend(pcm16_24k)

                # Encode as base64 and send to OpenAI
                audio_b64 = base64.b64encode(pcm16_24k).decode("ascii")
                await conn.input_audio_buffer.append(audio=audio_b64)

        except WebSocketDisconnect:
            logger.debug("Twilio WebSocket disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error forwarding user audio: {e}", exc_info=True)

    # ── OpenAI event processing ───────────────────────────────────────

    async def _process_openai_events(self, conn: Any, websocket: WebSocket) -> None:
        """Process events from the OpenAI Realtime connection."""
        try:
            async for event in conn:
                try:
                    await self._handle_openai_event(event, conn, websocket)
                except Exception as e:
                    logger.error(f"Error handling event {getattr(event, 'type', '?')}: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in OpenAI event loop: {e}", exc_info=True)

    async def _handle_openai_event(self, event: Any, conn: Any, websocket: WebSocket) -> None:
        """Dispatch a single OpenAI Realtime event."""
        event_type = getattr(event, "type", "")

        match event_type:
            case "session.created":
                logger.info("OpenAI Realtime session created")

            case "session.updated":
                logger.debug("OpenAI Realtime session updated")

            case "input_audio_buffer.speech_started":
                await self._on_speech_started(event)

            case "input_audio_buffer.speech_stopped":
                await self._on_speech_stopped(event)

            case "conversation.item.input_audio_transcription.completed":
                await self._on_transcription_completed(event)

            case "conversation.item.input_audio_transcription.delta":
                logger.debug(f"Transcription delta: {getattr(event, 'delta', '')}")

            case "conversation.item.input_audio_transcription.failed":
                error_info = getattr(event, "error", "")
                logger.warning(f"Transcription failed: {error_info}")
                # Gracefully handle transcription failure (e.g. API key lacks
                # whisper-1 access).  If a user turn was active but has no
                # transcript yet, record a placeholder so the turn is not lost.
                if self._user_turn and not self._user_turn.flushed:
                    timestamp_ms = self._user_turn.speech_started_wall_ms or None
                    self.audit_log.append_user_input(
                        "[user speech - transcription unavailable]",
                        timestamp_ms=timestamp_ms,
                    )
                    self._user_turn.flushed = True

            case "response.audio.delta":
                await self._on_audio_delta(event, websocket)

            case "response.audio_transcript.delta":
                self._on_transcript_delta(event)

            case "response.audio_transcript.done":
                self._on_transcript_done(event)

            case "response.function_call_arguments.done":
                await self._on_function_call_done(event, conn)

            case "response.done":
                await self._on_response_done(event)

            case "error":
                error_data = getattr(event, "error", None)
                logger.error(f"OpenAI Realtime error: {error_data}")

            case _:
                logger.debug(f"Unhandled OpenAI event: {event_type}")

    # ── Event handlers ────────────────────────────────────────────────

    async def _on_speech_started(self, event: Any) -> None:
        """Handle input_audio_buffer.speech_started."""
        wall = _wall_ms()

        # If assistant was responding, flush interrupted response
        if self._assistant_state.responding and self._assistant_state.transcript_parts:
            partial_text = "".join(self._assistant_state.transcript_parts) + " [interrupted]"
            self.audit_log.append_assistant_output(
                partial_text,
                timestamp_ms=self._assistant_state.first_audio_wall_ms,
            )
            if self._fw_log:
                self._fw_log.tts_text(partial_text)
                self._fw_log.turn_end(was_interrupted=True)
            logger.debug(f"Flushed interrupted assistant response: {partial_text[:60]}...")
            self._assistant_state = _AssistantResponseState()

        # Start new user turn
        self._user_turn = _UserTurnRecord(speech_started_wall_ms=wall)
        if self._fw_log:
            self._fw_log.turn_start(timestamp_ms=int(wall))

        logger.debug(f"Speech started at {wall}")

    async def _on_speech_stopped(self, event: Any) -> None:
        """Handle input_audio_buffer.speech_stopped."""
        wall = _wall_ms()
        if self._user_turn:
            self._user_turn.speech_stopped_wall_ms = wall
        else:
            self._user_turn = _UserTurnRecord(speech_stopped_wall_ms=wall)

        # Record latency measurement start
        self._speech_stopped_time = time.time()
        logger.debug(f"Speech stopped at {wall}")

    async def _on_transcription_completed(self, event: Any) -> None:
        """Handle conversation.item.input_audio_transcription.completed."""
        transcript = getattr(event, "transcript", "") or ""
        transcript = transcript.strip()

        if not transcript:
            logger.debug("Empty transcription, skipping")
            return

        timestamp_ms = None
        if self._user_turn:
            timestamp_ms = self._user_turn.speech_started_wall_ms or None
            self._user_turn.transcript = transcript
            self._user_turn.flushed = True

        self.audit_log.append_user_input(transcript, timestamp_ms=timestamp_ms)
        logger.debug(f"User transcription: {transcript}...")

    async def _on_audio_delta(self, event: Any, websocket: WebSocket) -> None:
        """Handle response.audio.delta - assistant audio chunk."""
        delta_b64 = getattr(event, "delta", "") or ""
        if not delta_b64:
            return

        # Decode base64 PCM16 audio
        pcm16_bytes = base64.b64decode(delta_b64)
        now = time.time()

        # --- Wall-clock aligned audio recording ---
        # Only insert silence between RESPONSES (not between delta chunks
        # within a single response).  This keeps the assistant audio track
        # aligned with the user track without inflating it with micro-gaps.
        if self._assistant_state.first_audio_wall_ms is None:
            # First audio chunk in a NEW response → pad silence from the
            # end of the previous response (or session start).
            ref_time = (
                self._assistant_audio_last_wall if self._assistant_audio_last_wall > 0 else self._session_start_wall
            )
            gap = now - ref_time
            if gap > 0.02:  # >20ms gap → insert silence
                silence_samples = int(gap * OPENAI_SAMPLE_RATE)
                silence_duration = silence_samples / OPENAI_SAMPLE_RATE
                logger.info(
                    f"[SILENCE DEBUG] Inserting {silence_duration:.3f}s silence "
                    f"({silence_samples} samples). gap={gap:.3f}s, "
                    f"ref_time={ref_time:.3f}, now={now:.3f}, "
                    f"asst_buf_before={len(self.assistant_audio_buffer)}, "
                    f"user_buf={len(self.user_audio_buffer)}"
                )
                self.assistant_audio_buffer.extend(b"\x00\x00" * silence_samples)
            else:
                logger.info(
                    f"[SILENCE DEBUG] Gap too small, NO silence inserted. "
                    f"gap={gap:.3f}s, ref_time={ref_time:.3f}, now={now:.3f}"
                )

            self._assistant_state.first_audio_wall_ms = _wall_ms()
            self._assistant_state.responding = True

            # Measure response latency (speech_stopped -> first audio)
            if hasattr(self, "_speech_stopped_time"):
                latency = now - self._speech_stopped_time
                self._latency_measurements.append(latency)
                if self._metrics_log:
                    self._metrics_log.write_ttfb_metric(
                        processor="openai_realtime",
                        value_seconds=latency,
                        model=self._model,
                    )
                logger.debug(f"Response latency: {latency:.3f}s")

        # Record assistant audio (no silence padding between delta chunks
        # within the same response — they arrive rapidly and represent
        # continuous speech).
        self.assistant_audio_buffer.extend(pcm16_bytes)
        self._last_audio_delta_wall = time.time()

        # Update the last-write wall time only at RESPONSE boundaries
        # (done in _reset_assistant_state, called from _on_response_done).

        # Convert 24kHz PCM16 -> 8kHz mulaw and send in small chunks
        # (160 bytes = 20ms at 8kHz mulaw) for proper user sim timing
        try:
            mulaw_bytes = pcm16_24k_to_mulaw_8k(pcm16_bytes)
            _MULAW_CHUNK = 160
            offset = 0
            while offset < len(mulaw_bytes):
                chunk = mulaw_bytes[offset : offset + _MULAW_CHUNK]
                offset += _MULAW_CHUNK
                twilio_msg = create_twilio_media_message(self._stream_sid, chunk)
                await websocket.send_text(twilio_msg)
        except Exception as e:
            logger.error(f"Error sending audio to Twilio WS: {e}")

    def _on_transcript_delta(self, event: Any) -> None:
        """Handle response.audio_transcript.delta - incremental assistant text."""
        delta = getattr(event, "delta", "") or ""
        if delta:
            self._assistant_state.transcript_parts.append(delta)

    def _on_transcript_done(self, event: Any) -> None:
        """Handle response.audio_transcript.done - full assistant transcript.

        This is the most reliable source of what the model actually said.
        Store it so _on_response_done can use it if delta accumulation failed.
        """
        transcript = getattr(event, "transcript", "") or ""
        if transcript:
            self._assistant_state.transcript_done_text = transcript.strip()
            logger.debug(f"Assistant transcript done: {transcript}...")
            if self._fw_log:
                self._fw_log.tts_text(transcript)

    async def _on_function_call_done(self, event: Any, conn: Any) -> None:
        """Handle response.function_call_arguments.done - execute tool call."""
        call_id = getattr(event, "call_id", "")
        func_name = getattr(event, "name", "")
        arguments_str = getattr(event, "arguments", "{}") or "{}"

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        logger.info(f"Tool call: {func_name}({json.dumps(arguments)})")
        self._assistant_state.has_function_calls = True

        # Record in audit log
        self.audit_log.append_realtime_tool_call(func_name, arguments)

        # Execute tool
        result = await self.tool_handler.execute(func_name, arguments)

        # Record tool response
        self.audit_log.append_tool_response(func_name, result)

        if self._fw_log:
            self._fw_log.write(
                "tool_call",
                {
                    "frame": "tool_call",
                    "tool_name": func_name,
                    "arguments": arguments,
                    "result": result,
                },
            )

        # Send function call output back to OpenAI
        await conn.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
        )

        # Trigger next response after tool result
        await conn.response.create()

    async def _on_response_done(self, event: Any) -> None:
        """Handle response.done - assistant response complete.

        Following the pipecat InstrumentedRealtimeLLMService pattern:
        - Only call append_assistant_output() (no append_llm_call)
        - Token usage goes to pipecat_metrics.jsonl only
        """
        # Extract usage metrics
        response = getattr(event, "response", None)
        if response:
            usage = getattr(response, "usage", None)
            if usage and self._metrics_log:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                self._metrics_log.write_token_usage(
                    processor="openai_realtime",
                    model=self._model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )

        has_function_calls = self._response_has_function_calls(event)

        # Build transcript text from best available source:
        # 1. response.audio_transcript.done text (most reliable)
        # 2. Accumulated response.audio_transcript.delta parts
        # 3. Text extracted from response.done output items
        content = self._assistant_state.transcript_done_text
        if not content:
            content = "".join(self._assistant_state.transcript_parts).strip()
        if not content:
            content = self._extract_response_text(event)

        audio_was_streamed = self._assistant_state.first_audio_wall_ms is not None

        # Skip tool-call-only responses (nothing spoken)
        if not content and has_function_calls:
            logger.debug("response_done: tool-call-only response, skipping assistant entry")
            self._reset_assistant_state()
            return

        # Skip mixed responses where audio was not streamed
        if content and not audio_was_streamed and has_function_calls:
            logger.debug(f"response_done: mixed response with no audio, skipping: '{content[:60]}...'")
            self._reset_assistant_state()
            return

        # If audio was streamed but we have no transcript at all, skip rather
        # than pollute the audit log with a placeholder.  The audio recording
        # still captures what was said.
        if not content and audio_was_streamed:
            logger.debug("response_done: audio streamed but no transcript available, skipping text entry")
            self._reset_assistant_state()
            return

        if not content:
            # No audio, no text, no function calls — nothing to log
            self._reset_assistant_state()
            return

        # Log assistant output (single entry — no append_llm_call)
        timestamp = self._assistant_state.first_audio_wall_ms or _wall_ms()
        self.audit_log.append_assistant_output(content, timestamp_ms=timestamp)

        if self._fw_log:
            self._fw_log.llm_response(content)
            self._fw_log.turn_end(was_interrupted=False)

        logger.debug(f"response_done: '{content[:60]}...'")
        self._reset_assistant_state()

    # ── Helpers ───────────────────────────────────────────────────────

    def _reset_assistant_state(self) -> None:
        """Clear accumulated assistant response state.

        Only updates the wall-clock reference when audio was actually
        streamed in this response. Tool-call-only responses must NOT
        advance the reference, otherwise the silence gap for the next
        audio response gets swallowed.
        """
        audio_was_streamed = self._assistant_state.first_audio_wall_ms is not None
        now = time.time()
        last_delta = getattr(self, "_last_audio_delta_wall", 0.0)
        drift = now - last_delta if last_delta > 0 else 0.0
        logger.info(
            f"[SILENCE DEBUG] reset_assistant_state: audio_was_streamed={audio_was_streamed}, "
            f"now={now:.3f}, last_audio_delta={last_delta:.3f}, drift={drift:.3f}s, "
            f"asst_buf={len(self.assistant_audio_buffer)}, "
            f"user_buf={len(self.user_audio_buffer)}"
        )
        if audio_was_streamed:
            self._assistant_audio_last_wall = now
        self._assistant_state = _AssistantResponseState()

    def _build_realtime_tools(self) -> list[dict]:
        """Convert agent tools to OpenAI Realtime session tool format.

        The Realtime API session.tools expects a flat structure:
        {type, name, description, parameters: {type, properties, required}}
        """
        tools: list[dict] = []
        if not self.agent.tools:
            return tools

        for tool in self.agent.tools:
            tools.append(
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
            )
        return tools

    @staticmethod
    def _response_has_function_calls(event: Any) -> bool:
        """Return True if the response.done event contains function_call outputs."""
        response = getattr(event, "response", None)
        if not response:
            return False
        output_items = getattr(response, "output", None) or []
        return any(getattr(item, "type", "") == "function_call" for item in output_items)

    @staticmethod
    def _extract_response_text(event: Any) -> str:
        """Extract text content from response.done output items."""
        response = getattr(event, "response", None)
        if not response:
            return ""

        output_items = getattr(response, "output", None) or []
        text_parts: list[str] = []

        for item in output_items:
            content_list = getattr(item, "content", None) or []
            for part in content_list:
                part_type = getattr(part, "type", "")
                if part_type in ("audio", "text"):
                    transcript = getattr(part, "transcript", None) or getattr(part, "text", None) or ""
                    if transcript:
                        text_parts.append(transcript)

        return "".join(text_parts).strip()
