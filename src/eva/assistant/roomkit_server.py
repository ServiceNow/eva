"""RoomKit-based assistant server for voice conversations.

Uses RoomKit's VoiceChannel for the voice pipeline (STT, TTS, VAD,
turn detection, audio recording) while reusing EVA's AgenticSystem,
AuditLog, and ToolExecutor for LLM reasoning and tool execution.
"""

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from roomkit import HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.twilio_ws import TwilioWebSocketBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.agentic.system import AgenticSystem
from eva.assistant.base import AssistantServerBase
from eva.assistant.services.llm import LiteLLMClient
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.models.config import AudioLLMConfig, PipelineConfig, SpeechToSpeechConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 24000
INITIAL_MESSAGE = "Hello! How can I help you today?"


# ---------------------------------------------------------------------------
# STT/TTS provider factories
# ---------------------------------------------------------------------------

_STT_PROVIDERS: dict[str, str] = {
    "deepgram": "roomkit.voice.stt.deepgram",
    "openai_whisper": "roomkit.voice.stt.openai",
}

_TTS_PROVIDERS: dict[str, str] = {
    "elevenlabs": "roomkit.voice.tts.elevenlabs",
}


def _create_stt_provider(config: PipelineConfig) -> Any:
    """Create a RoomKit STT provider from EVA's pipeline config."""
    name = config.stt or "deepgram"

    if name not in _STT_PROVIDERS:
        raise ValueError(f"Unsupported STT for RoomKit: {name}. Available: {sorted(_STT_PROVIDERS)}")

    if name == "deepgram":
        from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        return DeepgramSTTProvider(
            config=DeepgramConfig(
                api_key=api_key,
                model="nova-2",
                language="en",
                punctuate=True,
                smart_format=True,
                endpointing=300,
            )
        )

    raise ValueError(f"STT provider '{name}' registered but not implemented")


def _create_tts_provider(config: PipelineConfig) -> Any:
    """Create a RoomKit TTS provider from EVA's pipeline config."""
    name = config.tts or "elevenlabs"

    if name not in _TTS_PROVIDERS:
        raise ValueError(f"Unsupported TTS for RoomKit: {name}. Available: {sorted(_TTS_PROVIDERS)}")

    if name == "elevenlabs":
        from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        return ElevenLabsTTSProvider(
            config=ElevenLabsConfig(
                api_key=api_key,
                voice_id="21m00Tcm4TlvDq8ikWAM",
                model_id="eleven_multilingual_v2",
                output_format=f"pcm_{SAMPLE_RATE}",
                optimize_streaming_latency=3,
            )
        )

    raise ValueError(f"TTS provider '{name}' registered but not implemented")


# ---------------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------------


class RoomKitAssistantServer(AssistantServerBase):
    """RoomKit-based implementation of the assistant server.

    Uses RoomKit's VoiceChannel for the voice pipeline (STT, TTS, VAD,
    turn detection, audio recording) and EVA's AgenticSystem for LLM
    reasoning and tools. Currently supports PipelineConfig (STT+LLM+TTS)
    mode only.
    """

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
    ) -> None:
        if not isinstance(pipeline_config, PipelineConfig):
            raise NotImplementedError(
                "RoomKit server currently supports PipelineConfig (STT+LLM+TTS) only. "
                "SpeechToSpeech and AudioLLM modes are not yet implemented."
            )

        self.pipeline_config = pipeline_config
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.port = port
        self.conversation_id = conversation_id
        self.current_date_time = current_date_time

        # --- EVA components (Pipecat-free, reusable) ---
        self.audit_log = AuditLog()
        self.tool_handler = ToolExecutor(
            tool_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            tool_module_path=agent.tool_module_path,
            current_date_time=current_date_time,
        )
        llm_client = LiteLLMClient(model=pipeline_config.llm)
        self.agentic_system = AgenticSystem(
            current_date_time=current_date_time,
            agent=agent,
            tool_handler=self.tool_handler,
            audit_log=self.audit_log,
            llm_client=llm_client,
            output_dir=output_dir,
        )

        # --- RoomKit components ---
        self.kit = RoomKit()
        self.backend = TwilioWebSocketBackend(output_sample_rate=SAMPLE_RATE)
        stt = _create_stt_provider(pipeline_config)
        tts = _create_tts_provider(pipeline_config)

        # Audio recording via RoomKit's WavFileRecorder
        self._recorder = WavFileRecorder()
        self._recording_config = RecordingConfig(
            channels=RecordingChannelMode.ALL,
            storage=str(self.output_dir),
        )

        pipeline = AudioPipelineConfig(
            recorder=self._recorder,
            recording_config=self._recording_config,
        )

        self.voice = VoiceChannel(
            "voice",
            stt=stt,
            tts=tts,
            backend=self.backend,
            pipeline=pipeline,
            enable_barge_in=False,
        )
        self.kit.register_channel(self.voice)

        # --- Server state ---
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self._running = False
        self._session = None
        self._current_query_task: asyncio.Task | None = None

        # --- Framework logging ---
        self._framework_logs: list[dict] = []
        self._latency_measurements: list[float] = []
        self._was_interrupted = False
        self._user_speech_end_ts: float | None = None

    # -----------------------------------------------------------------------
    # AssistantServerBase interface
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._register_hooks()

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
            self._app, host="0.0.0.0", port=self.port, log_level="warning", lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"RoomKit assistant server started on ws://localhost:{self.port}")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._current_query_task and not self._current_query_task.done():
            self._current_query_task.cancel()

        try:
            await self.kit.close()
        except Exception as e:
            logger.warning(f"Error closing RoomKit: {e}")

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    if self._server_task:
                        self._server_task.cancel()
                        try:
                            await self._server_task
                        except asyncio.CancelledError:
                            pass
            self._server = None
            self._server_task = None

        await self._save_outputs()
        logger.info(f"RoomKit assistant server stopped on port {self.port}")

    def get_conversation_stats(self) -> dict[str, Any]:
        return self.audit_log.get_stats()

    # -----------------------------------------------------------------------
    # RoomKit hooks
    # -----------------------------------------------------------------------

    def _register_hooks(self) -> None:
        @self.kit.hook(HookTrigger.ON_SPEECH_START)
        async def on_speech_start(event, ctx):
            self._was_interrupted = False
            self._log_event("turn_start", {"frame": ""})
            return HookResult.allow()

        @self.kit.hook(HookTrigger.ON_SPEECH_END)
        async def on_speech_end(event, ctx):
            self._user_speech_end_ts = time.time()
            self._log_event("turn_end", {"frame": "", "was_interrupted": self._was_interrupted})
            return HookResult.allow()

        @self.kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event, ctx):
            text = event.text if hasattr(event, "text") else str(event)
            if not text or not text.strip():
                return HookResult.allow()

            logger.info(f"User said: {text}")
            self._save_transcript_entry("user", text)

            # Cancel any in-progress query before starting a new one
            if self._current_query_task and not self._current_query_task.done():
                self._current_query_task.cancel()
                try:
                    await self._current_query_task
                except asyncio.CancelledError:
                    pass

            # AgenticSystem handles audit_log internally
            self._current_query_task = asyncio.create_task(self._process_query(text))
            return HookResult.allow()

        @self.kit.hook(HookTrigger.BEFORE_TTS)
        async def before_tts(text, ctx):
            if self._user_speech_end_ts:
                latency = time.time() - self._user_speech_end_ts
                self._latency_measurements.append(latency)
                self._user_speech_end_ts = None
            self._log_event("tts_text", {"frame": text})
            return HookResult.allow()

        @self.kit.hook(HookTrigger.ON_BARGE_IN)
        async def on_barge_in(event, ctx):
            self._was_interrupted = True
            if self._current_query_task and not self._current_query_task.done():
                self._current_query_task.cancel()
            return HookResult.allow()

    # -----------------------------------------------------------------------
    # Session handling
    # -----------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:
        logger.info("Client connected to RoomKit assistant server")
        try:
            room = await self.kit.create_room(room_id=self.conversation_id)
            await self.kit.attach_channel(room.id, "voice")
            session = await self.backend.connect(room.id, "user", "voice")
            self._session = session
            self.backend.bind_websocket(websocket)
            await self.kit.join(room.id, "voice", session=session)
            self.backend.notify_session_ready(session)

            # Recording starts automatically via pipeline.on_session_started()

            # Send greeting concurrently with WebSocket loop. The backend's
            # write queue ensures send_json() never blocks receive_json().
            async def _greet():
                await self.voice.say(session, INITIAL_MESSAGE)
                self._save_transcript_entry("assistant", INITIAL_MESSAGE)
                self._log_event("tts_text", {"frame": INITIAL_MESSAGE})

            greeting_task = asyncio.create_task(_greet())
            await self._websocket_loop(websocket, session)
            if not greeting_task.done():
                greeting_task.cancel()

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
        finally:
            # Stop recording — pipeline manages the handle
            self._recorder.close()

            if self._session:
                try:
                    await self.backend.disconnect(self._session)
                except Exception:
                    pass

            if self.agentic_system:
                try:
                    self.agentic_system.save_agent_perf_stats()
                except Exception as e:
                    logger.error(f"Error saving agent perf stats: {e}")

            self._save_latencies()
            logger.info("Session ended")

    async def _websocket_loop(self, websocket: WebSocket, session) -> None:
        """Read Twilio-protocol messages and feed audio to RoomKit.

        Uses a dedicated reader task + queue so that inbound reads never
        compete with outbound sends (greeting TTS / assistant responses)
        for event loop time. Without this, outbound send_json() calls
        starve the receive side, causing inbound frames to buffer in TCP
        and arrive as bursts — producing choppy audio in recordings.
        """
        audio_queue: asyncio.Queue = asyncio.Queue()

        async def _reader():
            """Dedicated WebSocket reader — never yields to send operations."""
            try:
                while self._running:
                    try:
                        data = await websocket.receive_json()
                    except WebSocketDisconnect:
                        break
                    event_type = data.get("event")
                    if event_type == "media":
                        payload = data.get("media", {}).get("payload", "")
                        if payload:
                            await audio_queue.put(payload)
                    elif event_type == "stop":
                        logger.info("Twilio stream stopped")
                        break
            finally:
                await audio_queue.put(None)  # sentinel

        reader_task = asyncio.create_task(_reader())
        try:
            while True:
                payload = await audio_queue.get()
                if payload is None:
                    break
                await self.backend.feed_twilio_audio(session, payload)
        finally:
            if not reader_task.done():
                reader_task.cancel()

    # -----------------------------------------------------------------------
    # AI processing
    # -----------------------------------------------------------------------

    async def _process_query(self, text: str) -> None:
        """Run AgenticSystem and speak the response via RoomKit TTS."""
        if not self._session:
            return
        try:
            async for response in self.agentic_system.process_query(text):
                if response:
                    self._log_event("llm_response", {"frame": response})
                    self._save_transcript_entry("assistant", response)
                    await self.voice.say(self._session, response)
        except asyncio.CancelledError:
            logger.info("Query cancelled (user interrupted)")
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)

    # -----------------------------------------------------------------------
    # Logging and output
    # -----------------------------------------------------------------------

    def _log_event(self, event_type: str, data: dict) -> None:
        self._framework_logs.append({
            "type": event_type,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "conversation_id": self.conversation_id,
        })

    def _save_transcript_entry(self, role: str, content: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        try:
            with open(self.output_dir / "transcript.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")

    def _save_latencies(self) -> None:
        latencies = self._latency_measurements
        try:
            with open(self.output_dir / "response_latencies.json", "w") as f:
                json.dump({
                    "latencies": latencies,
                    "mean": sum(latencies) / len(latencies) if latencies else 0.0,
                    "max": max(latencies) if latencies else 0.0,
                    "count": len(latencies),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving latencies: {e}")

    async def _save_outputs(self) -> None:
        # Audit log
        self.audit_log.save(self.output_dir / "audit_log.json")

        # Transcript fallback
        transcript_path = self.output_dir / "transcript.jsonl"
        if not transcript_path.exists():
            self.audit_log.save_transcript_jsonl(transcript_path)

        # Scenario DB snapshots (REQUIRED)
        try:
            with open(self.output_dir / "initial_scenario_db.json", "w") as f:
                json.dump(self.tool_handler.original_db, f, indent=2, sort_keys=True, default=str)
            with open(self.output_dir / "final_scenario_db.json", "w") as f:
                json.dump(self.tool_handler.db, f, indent=2, sort_keys=True, default=str)
        except Exception as e:
            logger.error(f"Error saving scenario DBs: {e}", exc_info=True)
            raise

        # Audio: copy RoomKit's WAV recordings to EVA's expected filenames
        self._copy_recordings()

        # Framework logs
        with open(self.output_dir / "framework_logs.jsonl", "w") as f:
            for entry in self._framework_logs:
                f.write(json.dumps(entry) + "\n")

        # ElevenLabs events stub (required by metrics processor)
        self._write_elevenlabs_stub()

        logger.info(f"Outputs saved to {self.output_dir}")

    def _copy_recordings(self) -> None:
        """Rename RoomKit's WAV recordings to EVA's expected filenames."""
        # ALL mode produces: *_inbound.wav, *_outbound.wav, *_mixed.wav
        name_map = {"inbound": "audio_user.wav", "outbound": "audio_assistant.wav", "mixed": "audio_mixed.wav"}
        found = 0
        for wav_file in self.output_dir.glob("*.wav"):
            for suffix, eva_name in name_map.items():
                if wav_file.name.endswith(f"_{suffix}.wav"):
                    shutil.move(str(wav_file), self.output_dir / eva_name)
                    logger.info(f"Audio: {wav_file.name} -> {eva_name}")
                    found += 1
                    break
        if found == 0:
            logger.warning("No recording WAV files found in output directory")

    def _write_elevenlabs_stub(self) -> None:
        """Write a minimal elevenlabs_events.jsonl for metrics compatibility."""
        stub_path = self.output_dir / "elevenlabs_events.jsonl"
        with open(stub_path, "w") as f:
            f.write(json.dumps({
                "type": "connection_state",
                "timestamp": int(time.time() * 1000),
                "sequence": 1,
                "data": {"state": "ended", "details": {"reason": "goodbye"}},
            }) + "\n")
