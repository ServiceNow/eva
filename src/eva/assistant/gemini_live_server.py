"""Gemini Live AssistantServer for EVA-Bench.

Bridges between Twilio-framed WebSocket (user simulator) and Google's Gemini Live
API via the google-genai Python SDK.  Audio flows:

    User simulator (8 kHz mulaw)
        -> 16 kHz PCM16 -> Gemini Live input
    Gemini Live output (24 kHz PCM16)
        -> 8 kHz mulaw -> User simulator

All tool calls are executed locally via ToolExecutor; transcription events
from Gemini populate the audit log.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from google import genai
from google.genai import types

from eva.assistant.agentic.audit_log import current_timestamp_ms
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
from eva.assistant.base_server import INITIAL_MESSAGE, AbstractAssistantServer
from eva.models.agents import AgentConfig, AgentTool
from eva.models.config import PipelineConfig, SpeechToSpeechConfig, AudioLLMConfig
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# Default recording sample rate (Gemini outputs 24 kHz PCM)
_RECORDING_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

def _json_schema_type(python_type: str) -> str:
    """Map Python/EVA type names to JSON Schema / Gemini type strings."""
    mapping = {
        "string": "STRING",
        "str": "STRING",
        "integer": "INTEGER",
        "int": "INTEGER",
        "number": "NUMBER",
        "float": "NUMBER",
        "boolean": "BOOLEAN",
        "bool": "BOOLEAN",
        "array": "ARRAY",
        "list": "ARRAY",
        "object": "OBJECT",
        "dict": "OBJECT",
    }
    return mapping.get(python_type.lower(), "STRING")


def _convert_schema_properties(props: dict[str, Any]) -> dict[str, types.Schema]:
    """Recursively convert JSON Schema property dicts to Gemini Schema objects."""
    result: dict[str, types.Schema] = {}
    for name, defn in props.items():
        if not isinstance(defn, dict):
            result[name] = types.Schema(type="STRING")
            continue

        schema_type = _json_schema_type(defn.get("type", "string"))
        kwargs: dict[str, Any] = {"type": schema_type}

        if "description" in defn:
            kwargs["description"] = defn["description"]
        if "enum" in defn:
            kwargs["enum"] = defn["enum"]

        # Nested object
        if schema_type == "OBJECT" and "properties" in defn:
            kwargs["properties"] = _convert_schema_properties(defn["properties"])

        # Array items
        if schema_type == "ARRAY" and "items" in defn:
            items = defn["items"]
            if isinstance(items, dict):
                item_type = _json_schema_type(items.get("type", "string"))
                item_kwargs: dict[str, Any] = {"type": item_type}
                if "properties" in items:
                    item_kwargs["properties"] = _convert_schema_properties(items["properties"])
                kwargs["items"] = types.Schema(**item_kwargs)
            else:
                kwargs["items"] = types.Schema(type="STRING")

        result[name] = types.Schema(**kwargs)
    return result


def _agent_tools_to_gemini(agent: AgentConfig) -> list[types.Tool] | None:
    """Convert EVA AgentConfig tools to Gemini FunctionDeclaration list."""
    if not agent.tools:
        return None

    declarations: list[types.FunctionDeclaration] = []
    for tool in agent.tools:
        properties = _convert_schema_properties(tool.get_parameter_properties())
        required = tool.get_required_param_names()

        params_schema = types.Schema(
            type="OBJECT",
            properties=properties,
            required=required if required else None,
        )

        declarations.append(
            types.FunctionDeclaration(
                name=tool.function_name,
                description=f"{tool.name}: {tool.description}",
                parameters=params_schema,
                behavior=types.Behavior.BLOCKING,
            )
        )

    if not declarations:
        return None
    return [types.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Gemini Live AssistantServer
# ---------------------------------------------------------------------------

class GeminiLiveAssistantServer(AbstractAssistantServer):
    """Bridges Twilio WebSocket <-> Gemini Live API for EVA-Bench evaluation."""

    def __init__(
        self,
        current_date_time: str,
        pipeline_config: PipelineConfig | SpeechToSpeechConfig | AudioLLMConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
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
        )

        # Recording sample rate (Gemini outputs 24 kHz)
        self._audio_sample_rate = _RECORDING_SAMPLE_RATE

        # Server state
        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        self._running = False

        # Gemini model name from s2s_params or default
        s2s_params: dict[str, Any] = {}
        if isinstance(self.pipeline_config, SpeechToSpeechConfig):
            s2s_params = self.pipeline_config.s2s_params or {}
        self._model = self.pipeline_config.s2s if isinstance(self.pipeline_config, SpeechToSpeechConfig) else s2s_params.get("model", "gemini-2.0-flash-live-001")
        self._voice = s2s_params.get("voice", "Kore")
        self._language_code = s2s_params.get("language_code", "en-US")

        # Build system prompt (same pattern as pipecat realtime)
        prompt_manager = PromptManager()
        self._system_prompt = prompt_manager.get_prompt(
            "realtime_agent.system_prompt",
            agent_personality=agent.description,
            agent_instructions=agent.instructions,
            datetime=self.current_date_time,
        )

        # Build Gemini tools
        self._gemini_tools = _agent_tools_to_gemini(agent)

        # Framework log writers
        self._fw_log: Optional[FrameworkLogWriter] = None
        self._metrics_log: Optional[MetricsLogWriter] = None

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

        logger.info(f"GeminiLive server started on ws://localhost:{self.port}")

    async def stop(self) -> None:
        """Stop the server, save outputs."""
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
        logger.info(f"GeminiLive server stopped on port {self.port}")

    # ------------------------------------------------------------------
    # Gemini client factory
    # ------------------------------------------------------------------

    def _create_genai_client(self) -> genai.Client:
        """Create a google-genai Client using Vertex AI or API key."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info("Using Gemini API key for authentication")
            return genai.Client(api_key=api_key)

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if project:
            logger.info(f"Using Vertex AI (project={project}, location={location})")
            return genai.Client(vertexai=True, project=project, location=location)

        # Fallback: let the SDK resolve credentials (e.g. ADC)
        logger.info("No explicit credentials; relying on google-genai default resolution")
        return genai.Client()

    # ------------------------------------------------------------------
    # Live session configuration
    # ------------------------------------------------------------------

    def _build_live_config(self) -> types.LiveConnectConfig:
        """Build the LiveConnectConfig for the Gemini session."""
        config_kwargs: dict[str, Any] = {
            "response_modalities": [types.Modality.AUDIO],
            "system_instruction": self._system_prompt,
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice,
                    )
                ),
                language_code=self._language_code,
            ),
            "realtime_input_config": types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    silence_duration_ms=200,
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
        }
        if self._gemini_tools:
            config_kwargs["tools"] = self._gemini_tools

        return types.LiveConnectConfig(**config_kwargs)

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Bridge a single Twilio WebSocket session with Gemini Live."""
        logger.info("Client connected to GeminiLive server")

        stream_sid: str = self.conversation_id
        client = self._create_genai_client()
        live_config = self._build_live_config()

        # Track Twilio stream state
        twilio_connected = True

        # Accumulate assistant speech text per turn
        _assistant_turn_text: list[str] = []
        _user_turn_text: list[str] = []

        _in_model_turn = False
        _user_speaking = False

        _user_speech_end_ts: Optional[float] = None
        _first_audio_in_turn = False

        try:
            async with client.aio.live.connect(
                model=self._model, config=live_config
            ) as session:
                logger.info(f"Gemini Live session connected (model={self._model})")

                # Trigger the initial greeting using realtime text input.
                # send_client_content with Content turns is not supported by
                # some Live models (e.g. gemini-3.1-flash-live-preview), but
                # send_realtime_input(text=...) works universally.
                await session.send_realtime_input(text=f"Please greet with: {INITIAL_MESSAGE}")
                self._fw_log.turn_start()

                # ----- Concurrent tasks -----
                async def _forward_user_audio() -> None:
                    """Read Twilio WS messages, convert audio, send to Gemini."""
                    nonlocal stream_sid, twilio_connected
                    try:
                        while twilio_connected and self._running:
                            try:
                                raw = await asyncio.wait_for(
                                    websocket.receive_text(), timeout=1.0
                                )
                            except asyncio.TimeoutError:
                                continue

                            # Parse Twilio JSON envelope
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            event = msg.get("event")
                            if event == "start":
                                stream_sid = msg.get("start", {}).get("streamSid", stream_sid)
                                logger.info(f"Twilio stream started: {stream_sid}")
                                continue
                            elif event == "stop":
                                logger.info("Twilio stream stopped")
                                twilio_connected = False
                                break
                            elif event == "media":
                                # Extract raw mulaw bytes
                                mulaw_bytes = parse_twilio_media_message(raw)
                                if mulaw_bytes is None:
                                    continue

                                # Convert 8 kHz mulaw -> 16 kHz PCM for Gemini
                                pcm_16k = mulaw_8k_to_pcm16_16k(mulaw_bytes)

                                pcm_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)
                                if not _in_model_turn:
                                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                                self.user_audio_buffer.extend(pcm_24k)

                                # Send to Gemini
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=pcm_16k,
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )
                    except WebSocketDisconnect:
                        logger.info("Twilio WebSocket disconnected")
                        twilio_connected = False
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in user audio forwarder: {e}", exc_info=True)
                    finally:
                        twilio_connected = False

                async def _process_gemini_events() -> None:
                    """Consume events from the Gemini Live session."""
                    nonlocal _assistant_turn_text, _user_turn_text
                    nonlocal _in_model_turn, _user_speaking, _user_speech_end_ts, _first_audio_in_turn
                    nonlocal twilio_connected

                    logger.info("Gemini event processor started")
                    event_count = 0
                    try:
                        # Use manual receive loop instead of `async for ... in session.receive()`
                        # because the iterator exits after turn_complete (returns None),
                        # closing the session prematurely. The manual loop keeps the session
                        # alive between model turns.
                        while self._running:
                            try:
                                response = await asyncio.wait_for(session._receive(), timeout=2.0)
                            except asyncio.TimeoutError:
                                continue
                            if response is None:
                                continue
                            if not self._running:
                                break

                            event_count += 1

                            # --- Server content (audio, transcriptions, turn signals) ---
                            if response.server_content:
                                sc = response.server_content

                                # Model audio output
                                if sc.model_turn:
                                    if not _in_model_turn:
                                        _in_model_turn = True
                                        _first_audio_in_turn = True
                                        _assistant_turn_text = []
                                        self._fw_log.turn_start()

                                    for part in sc.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            pcm_24k = bytes(part.inline_data.data)

                                            # Skip tiny chunks that can't be resampled
                                            if len(pcm_24k) < 6:
                                                continue

                                            # Measure latency: time from user speech end
                                            # to first assistant audio chunk
                                            if _first_audio_in_turn and _user_speech_end_ts is not None:
                                                latency = time.time() - _user_speech_end_ts
                                                self._latency_measurements.append(latency)
                                                self._metrics_log.write_ttfb_metric(
                                                    "gemini_live", latency, component="e2e_ttfb"
                                                )
                                                logger.debug(f"Response latency: {latency:.3f}s")
                                                _user_speech_end_ts = None
                                                _first_audio_in_turn = False

                                            if not _user_speaking:
                                                sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                                            self.assistant_audio_buffer.extend(pcm_24k)

                                            # Convert to 8 kHz mulaw and send in
                                            # small chunks so the user simulator's
                                            # silence-detection timing works correctly.
                                            if twilio_connected:
                                                try:
                                                    mulaw = pcm16_24k_to_mulaw_8k(pcm_24k)
                                                except Exception as conv_err:
                                                    logger.warning(f"Audio conversion error ({len(pcm_24k)} bytes): {conv_err}")
                                                    continue
                                                _MULAW_CHUNK = 160
                                                offset = 0
                                                while offset < len(mulaw):
                                                    chunk = mulaw[offset:offset + _MULAW_CHUNK]
                                                    offset += _MULAW_CHUNK
                                                    twilio_msg = create_twilio_media_message(
                                                        stream_sid, chunk
                                                    )
                                                    try:
                                                        await websocket.send_text(twilio_msg)
                                                    except Exception:
                                                        twilio_connected = False
                                                        break

                                # Turn complete
                                if sc.turn_complete:
                                    logger.debug("Gemini turn complete")
                                    full_text = " ".join(_assistant_turn_text).strip()
                                    if full_text:
                                        self.audit_log.append_assistant_output(full_text)
                                        self._fw_log.tts_text(full_text)
                                        self._fw_log.llm_response(full_text)
                                    self._fw_log.turn_end(was_interrupted=False)
                                    _in_model_turn = False
                                    _assistant_turn_text = []

                                # Barge-in / interruption
                                if sc.interrupted:
                                    _user_speaking = True
                                    logger.debug("Gemini turn interrupted (barge-in)")
                                    full_text = " ".join(_assistant_turn_text).strip()
                                    if full_text:
                                        self.audit_log.append_assistant_output(
                                            full_text + " [interrupted]"
                                        )
                                        self._fw_log.tts_text(full_text)
                                    self._fw_log.turn_end(was_interrupted=True)
                                    _in_model_turn = False
                                    _assistant_turn_text = []

                                # Input transcription (user speech)
                                if sc.input_transcription:
                                    _user_speaking = False
                                    text = sc.input_transcription.text or ""
                                    if text.strip():
                                        logger.info(f"User transcription: {text.strip()}")
                                        self.audit_log.append_user_input(text.strip())
                                        _user_speech_end_ts = time.time()

                                # Output transcription (model speech)
                                if sc.output_transcription:
                                    text = sc.output_transcription.text or ""
                                    if text.strip():
                                        _assistant_turn_text.append(text.strip())
                                        logger.debug(f"Assistant transcription chunk: {text.strip()}")

                            # --- Tool calls ---
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    tool_name = fc.name
                                    tool_args = dict(fc.args) if fc.args else {}
                                    logger.info(f"Tool call: {tool_name}({json.dumps(tool_args)})")

                                    # Record in audit log
                                    self.audit_log.append_realtime_tool_call(
                                        tool_name, tool_args
                                    )

                                    # Execute tool
                                    result = await self.tool_handler.execute(
                                        tool_name, tool_args
                                    )
                                    logger.info(f"Tool result: {tool_name} -> {json.dumps(result)}")
                                    self.audit_log.append_tool_response(tool_name, result)

                                    # Send result back to Gemini
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=result,
                                            )
                                        ]
                                    )

                            # --- Usage metadata ---
                            if response.usage_metadata:
                                um = response.usage_metadata
                                prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
                                completion_tokens = (
                                    getattr(um, "candidates_token_count", 0) or 0
                                )
                                if prompt_tokens or completion_tokens:
                                    self._metrics_log.write_token_usage(
                                        processor="gemini_live",
                                        model=self._model,
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens,
                                    )

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in Gemini event processor: {e}", exc_info=True)

                # Run both tasks; when either exits, cancel the other
                user_task = asyncio.create_task(_forward_user_audio())
                gemini_task = asyncio.create_task(_process_gemini_events())

                done, pending = await asyncio.wait(
                    [user_task, gemini_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Log which task finished first
                for task in done:
                    task_name = "user_audio" if task is user_task else "gemini_events"
                    exc = task.exception()
                    if exc:
                        logger.error(f"Task '{task_name}' failed: {exc}", exc_info=exc)
                    else:
                        logger.info(f"Task '{task_name}' completed normally")

                for task in pending:
                    task_name = "user_audio" if task is user_task else "gemini_events"
                    logger.info(f"Cancelling pending task '{task_name}'")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.error(f"Gemini Live session error: {e}", exc_info=True)
        finally:
            logger.info("Client disconnected from GeminiLive server")
