"""Assistant server - Pipecat-based WebSocket server for voice conversations.

This module provides the Pipecat pipeline server that the user simulator connects to.
It handles audio streaming via WebSocket with Twilio-style frame serialization.
"""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    CancelFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.time import time_now_iso8601

from eva.assistant.agentic.audit_log import convert_to_epoch_ms, current_timestamp_ms
from eva.assistant.base_server import INITIAL_MESSAGE, AbstractAssistantServer
from eva.assistant.pipeline.agent_processor import BenchmarkAgentProcessor, UserAudioCollector, UserObserver
from eva.assistant.pipeline.audio_llm_processor import (
    AudioLLMProcessor,
    AudioLLMUserAudioCollector,
    AudioTranscriptionProcessor,
    InputTranscriptionContextFilter,
)
from eva.assistant.pipeline.observers import BenchmarkLogObserver, MetricsFileObserver, WallClock
from eva.assistant.pipeline.realtime_llm import InstrumentedRealtimeLLMService
from eva.assistant.pipeline.services import (
    create_audio_llm_client,
    create_realtime_llm_service,
    create_stt_service,
    create_tts_service,
)
from eva.assistant.services.llm import LiteLLMClient
from eva.models.agents import AgentConfig
from eva.models.config import AudioLLMConfig, PipelineConfig, SpeechToSpeechConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Audio/VAD constants
SAMPLE_RATE = 24000
VAD_STOP_SECS = 0.2  # How long silence must be detected before triggering stop (pipecat default: 0.2)
SMART_TURN_STOP_SECS = 3  # Default from SmartTurnParams
# Pre-speech audio buffer - captures audio BEFORE VAD fires to avoid cutting off speech start.
# Should be larger than pipecat's VAD start_secs (0.2s) to account for VAD latency.
VAD_PRE_SPEECH_BUFFER_SECS = 0.5


class PipecatAssistantServer(AbstractAssistantServer):
    """Pipecat-based WebSocket server for the assistant in voice conversations.

    This server:
    - Accepts WebSocket connections from the user simulator
    - Uses Pipecat pipeline with STT → Agent → TTS
    - Handles Twilio-style frame serialization (compatible with ElevenLabs client)
    - Records all audio and transcripts
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
    ):
        """Initialize the assistant server.

        Args:
            current_date_time: Current date/time string from the evaluation record
            pipeline_config: Configuration for STT/TTS/LLM services
            agent: Single agent configuration to use
            agent_config_path: Path to agent YAML configuration
            scenario_db_path: Path to scenario database JSON
            output_dir: Directory for output files
            port: Port to listen on
            conversation_id: Unique ID for this conversation
        """
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

        self.agentic_system = None  # Will be set in _handle_session

        # Wall-clock captured at on_user_turn_started for non-instrumented S2S models
        self._user_turn_started_wall_ms: str | None = None

        # Override audio sample rate for pipecat
        self._audio_sample_rate = SAMPLE_RATE

        # Server state
        self._app = None
        self._server = None
        self._server_task = None
        self._runner: PipelineRunner | None = None
        self._task: PipelineTask | None = None
        self._running = False
        self.num_seconds = 0
        self._metrics_observer: MetricsFileObserver | None = None
        self.non_instrumented_realtime_llm = False

    async def start(self) -> None:
        """Start the FastAPI WebSocket server."""
        if self._running:
            logger.warning("Server already running")
            return

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create FastAPI app
        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        # Also support root path for compatibility
        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        # Start uvicorn server
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)

        # Run server in background task
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        # Wait for server to be ready
        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Assistant server started on ws://localhost:{self.port}")

    async def stop(self) -> None:
        """Stop the server and save outputs."""
        if not self._running:
            return

        self._running = False

        # Stop the pipeline task
        if self._task:
            await self._task.cancel()
            self._task = None

        # Stop the server gracefully
        if self._server:
            self._server.should_exit = True
            # Wait briefly for graceful shutdown, then cancel if needed
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    # Force cancellation if graceful shutdown times out
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass  # Expected
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass  # Expected during shutdown
            self._server = None
            self._server_task = None

        # Save outputs
        await self.save_outputs()

        logger.info(f"Assistant server stopped on port {self.port}")

    async def _handle_session(self, websocket) -> None:
        """Handle a WebSocket session with the Pipecat pipeline."""
        logger.info("Client connected to assistant server")

        try:
            # Create transport with Twilio serializer
            transport = self._create_transport(websocket)
            # Create services
            realtime_llm = None
            audio_llm_processor = None
            audio_llm_audio_collector = None
            alm_client = None
            if isinstance(self.pipeline_config, SpeechToSpeechConfig):
                realtime_llm = create_realtime_llm_service(
                    self.pipeline_config.s2s,
                    self.pipeline_config.s2s_params,
                    self.agent,
                    audit_log=self.audit_log,
                    current_date_time=self.current_date_time,
                )
                if not isinstance(realtime_llm, InstrumentedRealtimeLLMService):
                    self.non_instrumented_realtime_llm = True

                async def _realtime_tool_handler(params) -> None:
                    """Wraps tool execution to record calls/responses in the audit log."""
                    tool_name = params.function_name
                    raw_args = params.arguments
                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            arguments = {}
                    elif isinstance(raw_args, dict):
                        arguments = raw_args
                    else:
                        arguments = {}

                    result = await self.execute_tool(tool_name, arguments)
                    await params.result_callback(result)

                realtime_llm.register_function(function_name=None, handler=_realtime_tool_handler)
                stt = None
                tts = None
            elif isinstance(self.pipeline_config, AudioLLMConfig):
                # Audio-LLM mode: model handles STT+LLM, TTS still needed
                stt = None
                tts = create_tts_service(
                    self.pipeline_config.tts,
                    self.pipeline_config.tts_params,
                )

                alm_client = create_audio_llm_client(
                    self.pipeline_config.audio_llm,
                    self.pipeline_config.audio_llm_params,
                )
                # Note: audio_llm_audio_collector and audio_llm_processor are created
                # after context/user_aggregator below (they need those references)
            else:
                stt = create_stt_service(
                    self.pipeline_config.stt,
                    self.pipeline_config.stt_params,
                )
                tts = create_tts_service(
                    self.pipeline_config.tts,
                    self.pipeline_config.tts_params,
                )
                # Create LLM client for agentic system (separate from Pipecat LLM service)
                llm_client = LiteLLMClient(model=self.pipeline_config.llm)

            # Create context aggregator with user turn strategies
            messages = []
            if realtime_llm:
                messages = [{"role": "user", "content": f"Say '{INITIAL_MESSAGE}'"}]
            context = LLMContext(messages=messages)
            vad_stop_secs = VAD_STOP_SECS
            smart_turn_stop_secs = SMART_TURN_STOP_SECS
            if alm_client:
                TurnAnalyzerUserTurnStopStrategy._maybe_trigger_user_turn_stopped = (
                    override__maybe_trigger_user_turn_stopped
                )
                vad_stop_secs = self.pipeline_config.audio_llm_params.get(
                    "vad_stop_secs", 0.4
                )  # Longer silence default because we don't have an stt transcript
                smart_turn_stop_secs = self.pipeline_config.audio_llm_params.get(
                    "smart_turn_stop_secs", 0.8
                )  # Shorter silence so we don't have to wait 3s if smart turn marks audio as incomplete

            if (
                isinstance(self.pipeline_config, (PipelineConfig, SpeechToSpeechConfig))
                and self.pipeline_config.turn_strategy == "external"
            ):
                logger.info("Using external user turn strategies")
                user_turn_strategies = ExternalUserTurnStrategies()
                vad_analyzer = None
            else:
                logger.info("Using local smart turn analyzer")
                user_turn_strategies = UserTurnStrategies(
                    start=[VADUserTurnStartStrategy()],
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(
                            turn_analyzer=LocalSmartTurnAnalyzerV3(
                                params=SmartTurnParams(stop_secs=smart_turn_stop_secs)
                            )
                        )
                    ],
                )
                vad_analyzer = SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs))
            user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
                context,
                user_params=LLMUserAggregatorParams(
                    vad_analyzer=vad_analyzer,
                    user_turn_strategies=user_turn_strategies,
                ),
            )

            # Create Audio-LLM components now that context/user_aggregator are available
            audio_llm_audio_collector = None
            audio_llm_processor = None
            if isinstance(self.pipeline_config, AudioLLMConfig):
                assert alm_client is not None  # Set in AudioLLMConfig branch above
                audio_llm_audio_collector = AudioLLMUserAudioCollector(
                    context, user_aggregator, pre_speech_secs=VAD_PRE_SPEECH_BUFFER_SECS
                )
                audio_llm_processor = AudioLLMProcessor(
                    current_date_time=self.current_date_time,
                    agent=self.agent,
                    tool_handler=self.tool_handler,
                    audit_log=self.audit_log,
                    alm_client=alm_client,
                    audio_collector=audio_llm_audio_collector,
                    output_dir=self.output_dir,
                )
                audio_llm_processor.on_assistant_response = lambda msg: self._save_transcript_message_from_turn(
                    role="assistant", content=msg, timestamp=self._current_iso_timestamp()
                )
                self.agentic_system = audio_llm_processor.agentic_system

            # Create transcription components for parallel pipeline (used in Audio-LLM mode)
            input_transcription_context_filter = None
            input_transcription_processor = None
            if audio_llm_audio_collector is not None:
                input_transcription_context_filter = InputTranscriptionContextFilter()
                input_transcription_processor = AudioTranscriptionProcessor(
                    audio_collector=audio_llm_audio_collector,
                    model=self.pipeline_config.audio_llm_params.get("model"),
                    params=self.pipeline_config.audio_llm_params,
                    sample_rate=SAMPLE_RATE,
                )

                # Set callback to save user transcription to transcript.jsonl and update audit log
                async def on_user_transcription(text: str, timestamp: str, turn_id: int | None) -> None:
                    await self._save_transcript_message_from_turn(role="user", content=text, timestamp=timestamp)
                    # Update the audit log placeholder with real transcription
                    if turn_id is not None:
                        self.audit_log.update_user_input_by_turn_id(turn_id, text)
                    else:
                        # Fallback to last-entry update if no turn_id
                        self.audit_log.update_last_user_input(text)

                input_transcription_processor.on_transcription = on_user_transcription

            # Create audio collector (used when STT is disabled)
            audio_collector = UserAudioCollector(context, user_aggregator)

            # Create processors
            # Configure audio buffer with 1-second buffer size for event triggering
            audiobuffer = AudioBufferProcessor(
                sample_rate=SAMPLE_RATE,
                num_channels=1,  # Mono (mixed user + bot audio)
                buffer_size=SAMPLE_RATE * 2,  # 1 second of 16-bit audio (2 bytes per sample)
            )
            # Create agent processor (pipeline mode only — realtime handles LLM internally)
            agent_processor = None
            if isinstance(self.pipeline_config, PipelineConfig):
                agent_processor = BenchmarkAgentProcessor(
                    current_date_time=self.current_date_time,
                    agent=self.agent,
                    tool_handler=self.tool_handler,
                    audit_log=self.audit_log,
                    llm_client=llm_client,
                    output_dir=self.output_dir,
                )
                agent_processor.on_assistant_response = lambda msg: self._save_transcript_message_from_turn(
                    role="assistant", content=msg, timestamp=self._current_iso_timestamp()
                )
                self.agentic_system = agent_processor.agentic_system

            # Create pipeline
            pipeline = self._create_pipeline(
                transport=transport,
                stt=stt,
                tts=tts,
                realtime_llm=realtime_llm,
                user_aggregator=user_aggregator,
                assistant_aggregator=assistant_aggregator,
                audiobuffer=audiobuffer,
                agent_processor=agent_processor,
                audio_collector=audio_collector,
                audio_llm_processor=audio_llm_processor,
                audio_llm_audio_collector=audio_llm_audio_collector,
                input_transcription_context_filter=input_transcription_context_filter,
                input_transcription_processor=input_transcription_processor,
            )

            metrics_log_path = self.output_dir / "pipecat_metrics.jsonl"

            # Create wall clock for consistent timestamps across log sources
            wall_clock = WallClock()

            self._metrics_observer = MetricsFileObserver(metrics_log_path, clock=wall_clock)

            observers = [
                BenchmarkLogObserver(str(self.output_dir), self.conversation_id, clock=wall_clock),
                MetricsLogObserver(),  # Log all metrics to console
                self._metrics_observer,  # Write metrics to JSONL file
            ]

            self._task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True,  # Enable TTFB and ProcessingMetricsData
                    enable_usage_metrics=True,  # Enable LLM/TTS usage metrics
                ),
                clock=wall_clock,
                enable_turn_tracking=True,
                observers=observers,
            )

            # Setup event handlers
            self._setup_event_handlers(
                transport=transport,
                audiobuffer=audiobuffer,
                task=self._task,
                user_aggregator=user_aggregator,
                assistant_aggregator=assistant_aggregator,
                agent_processor=agent_processor,
                audio_llm_processor=audio_llm_processor,
                input_transcription_processor=input_transcription_processor,
            )

            # Run the pipeline
            self._runner = PipelineRunner(handle_sigint=False, force_gc=True)
            await self._runner.run(self._task)

        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
        finally:
            # Save agent performance stats before cleanup
            if self.agentic_system:
                try:
                    logger.info("Saving agent performance stats from finally block...")
                    self.agentic_system.save_agent_perf_stats()
                except Exception as e:
                    logger.error(f"Error saving agent perf stats in finally: {e}", exc_info=True)

            # Close the metrics file observer explicitly to flush and release the file handle
            if self._metrics_observer:
                self._metrics_observer.close()
                self._metrics_observer = None

            logger.info("Client disconnected from assistant server")

    def _create_transport(self, websocket) -> FastAPIWebsocketTransport:
        """Create the WebSocket transport with Twilio frame serialization."""
        return FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=TwilioFrameSerializer(
                    self.conversation_id,
                    params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
                ),
            ),
        )

    def _create_pipeline(
        self,
        transport,
        stt,
        tts,
        realtime_llm,
        user_aggregator,
        assistant_aggregator,
        audiobuffer,
        agent_processor,
        audio_collector,
        audio_llm_processor=None,
        audio_llm_audio_collector=None,
        input_transcription_context_filter=None,
        input_transcription_processor=None,
    ) -> Pipeline:
        """Create the Pipecat pipeline.

        Based on create_pipeline_tts() from chatbot.py.
        """
        pipeline_components = [transport.input()]

        if realtime_llm:
            stt = None
            tts = None

        # Add STT if configured, otherwise use audio collector for direct audio input
        if stt:
            pipeline_components.append(stt)
            # CRITICAL ORDER: Processors that need to SEE frames must come BEFORE user_aggregator
            # because user_aggregator CONSUMES frames (doesn't pass through)
            pipeline_components.append(UserObserver())  # For metrics
            pipeline_components.append(user_aggregator)  # Aggregates & fires turn events
            # Add agent processor (receives turn events via event handler)
            pipeline_components.append(agent_processor)
        elif audio_llm_processor:
            # Audio-LLM pipeline: collector buffers audio, processor handles turns
            pipeline_components.append(audio_llm_audio_collector)  # Buffers audio frames
            pipeline_components.append(UserObserver())  # For metrics
            pipeline_components.append(user_aggregator)  # Aggregates & fires turn events
            pipeline_components.append(
                ParallelPipeline(
                    [  # transcribe - OpenAI transcription branch
                        input_transcription_context_filter,
                        input_transcription_processor,
                    ],
                    [  # conversation inference - Audio-LLM branch
                        audio_llm_processor,
                    ],
                )
            )
        else:
            pipeline_components.append(audio_collector)
            pipeline_components.append(UserObserver())  # For metrics
            pipeline_components.append(user_aggregator)  # Aggregates & fires turn events
            pipeline_components.append(realtime_llm)

        # Add TTS if configured
        if tts:
            pipeline_components.append(tts)

        # Add output components (matches original chatbot.py order)
        pipeline_components.extend(
            [
                transport.output(),
                audiobuffer,
                assistant_aggregator,
            ]
        )

        logger.debug(f"Pipeline: {pipeline_components}")
        return Pipeline(pipeline_components)

    def _setup_event_handlers(
        self,
        transport,
        audiobuffer,
        task,
        user_aggregator,
        assistant_aggregator,
        agent_processor,
        audio_llm_processor=None,
        input_transcription_processor=None,
    ) -> None:
        """Setup event handlers for the pipeline."""

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipeline client connected")
            await audiobuffer.start_recording()

            # Send initial greeting
            if isinstance(self.pipeline_config, SpeechToSpeechConfig):
                await task.queue_frames([LLMRunFrame()])
            else:
                await task.queue_frames([TTSSpeakFrame(INITIAL_MESSAGE)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipeline client disconnected")
            await audiobuffer.stop_recording()
            await user_aggregator.reset()
            await assistant_aggregator.reset()
            await task.queue_frames([CancelFrame()])

        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            # Accumulate audio chunks
            self._audio_buffer.extend(audio)
            self._audio_sample_rate = sample_rate
            if len(audio) > 0:
                self.num_seconds += 1
                # helps to measure that the audio timing matches up to pipecat during the run.
                # when the logs say that silence is being sent, the audio duration should also increase by 1 second.
                logger.debug(f"Audio duration: {self.num_seconds} seconds")

        @audiobuffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(buffer, user_audio, bot_audio, sample_rate, num_channels):
            # Accumulate audio chunks
            self.user_audio_buffer.extend(user_audio)
            self.assistant_audio_buffer.extend(bot_audio)

        # Turn tracking
        if task.turn_tracking_observer:
            turn_observer = task.turn_tracking_observer

            @turn_observer.event_handler("on_turn_started")
            async def on_turn_started(observer, turn_number):
                logger.debug(f"Turn {turn_number} started")

            @turn_observer.event_handler("on_turn_ended")
            async def on_turn_ended(observer, turn_number, duration, was_interrupted):
                status = "interrupted" if was_interrupted else "completed"
                logger.debug(f"Turn {turn_number} {status} in {duration:.2f}s")

        # User turn events (for processing complete user transcripts)
        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
            """Process complete user turn from Pipecat's turn management.

            This fires when the user stops speaking and provides the complete
            transcript, preventing multiple LLM calls per user turn.
            """
            logger.info(f"User turn stopped - complete transcript: '{message.content}'")
            logger.debug(f"Turn timestamp: {message.timestamp}, user_id: {message.user_id}")

            if isinstance(self.pipeline_config, PipelineConfig):
                # STT provides real transcript text — save it now
                await self._save_transcript_message_from_turn(
                    role="user", content=message.content, timestamp=message.timestamp
                )
                await agent_processor.process_complete_user_turn(message.content)
            elif isinstance(self.pipeline_config, AudioLLMConfig) and audio_llm_processor:
                # No STT → message.content is empty.
                # Processing is triggered by LLMContextFrame flow through ParallelPipeline
                # (AudioLLMUserAudioCollector pushes LLMContextFrame on UserStoppedSpeakingFrame)
                pass
            elif self.non_instrumented_realtime_llm:
                # Non-instrumented realtime fallback (e.g. Ultravox)
                if message.content:
                    self.audit_log.append_user_input(
                        message.content,
                        timestamp_ms=self._user_turn_started_wall_ms,
                    )
                    self._user_turn_started_wall_ms = None
                    await self._save_transcript_message_from_turn(
                        role="user", content=message.content, timestamp=self._user_turn_started_wall_ms
                    )

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            logger.info(f"User turn started (strategy: {strategy.__class__.__name__})")
            # Capture wall-clock for non-instrumented S2S models so the audit log
            # timestamp reflects when the user actually started speaking.
            self._user_turn_started_wall_ms = current_timestamp_ms()

        # Assistant turn events (for logging)
        @assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            """Log when assistant turn completes."""
            logger.info(f"Assistant turn stopped - complete response: '{message.content}'")
            logger.debug(f"Turn timestamp: {message.timestamp}")

            if self.non_instrumented_realtime_llm:
                # Non-instrumented realtime fallback (e.g. Ultravox)
                # Prefer content from the aggregator (populated when output_modalities includes
                # "text").
                content = message.content
                self.audit_log.append_assistant_output(
                    content or "[audio response - transcription unavailable]",
                    timestamp_ms=convert_to_epoch_ms(message.timestamp),
                )
                await self._save_transcript_message_from_turn(
                    role="assistant", content=content, timestamp=convert_to_epoch_ms(message.timestamp)
                )

    async def _save_transcript_message_from_turn(self, role: str, content: str, timestamp: str) -> None:
        """Save a transcript message from aggregator turn events to JSONL file."""
        transcript_path = self.output_dir / "transcript.jsonl"

        log_entry = {
            "timestamp": timestamp,
            "role": role,
            "content": content,
        }

        try:
            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")

    @staticmethod
    def _current_iso_timestamp() -> str:
        """Return the current time as an ISO 8601 string with timezone."""
        return time_now_iso8601()

    async def save_outputs(self) -> None:
        """Save all outputs, with pipecat-specific additions."""
        # Save agent performance stats (pipecat-specific: AgenticSystem tracking)
        if self.agentic_system:
            try:
                logger.info("Saving agent performance stats from save_outputs()...")
                self.agentic_system.save_agent_perf_stats()
            except Exception as e:
                logger.error(f"Error saving agent perf stats: {e}", exc_info=True)

        # Call base class to save audit_log, audio, scenario DBs, latencies
        await super().save_outputs()


async def override__maybe_trigger_user_turn_stopped(self):
    """Trigger user turn stopped if conditions are met.

    Conditions:
    - Don't wait for transcript
    - Ensure turn analyzer indicates turn is complete
    - Ensure the timeout has elapsed
    """
    if not self._turn_complete:
        return

    # For non-finalized, only trigger if timeout task has completed
    if self._timeout_task is None:
        await self.trigger_user_turn_stopped()
