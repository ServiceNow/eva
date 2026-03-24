"""Frame processors for the Pipecat pipeline."""

import asyncio
import time
from collections.abc import Awaitable

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.agentic.system import AgenticSystem
from eva.assistant.pipeline.frames import (
    LLMMessageFrame,
    SpokenMessageFrame,
    TurnTimestampFrame,
    UserMessageFrame,
    VADBufferFrame,
)
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# VAD timing constants
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.8

# Frame types that require special websocket processing
WEBSOCKET_FRAME_TYPES = (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    SpokenMessageFrame,
    TranscriptionFrame,
    TurnTimestampFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)


class BenchmarkAgentProcessor(FrameProcessor):
    """Process incoming frames/text to match to an agent, allow agent to process and then forward to LLM."""

    def __init__(
        self,
        current_date_time: str,
        agent: AgentConfig,
        tool_handler: ToolExecutor,
        audit_log: AuditLog,
        llm_client,
        output_dir=None,
        **kwargs,
    ) -> None:
        """Initialize the agent processor.

        Args:
            current_date_time: Current date/time string from the evaluation record
            agent: Single agent configuration to use
            tool_handler: Handler for tool calls (ToolExecutor)
            audit_log: Audit log for conversation tracking
            llm_client: LLM client for generating responses
            output_dir: Optional output directory for saving performance stats
            **kwargs: Additional keyword arguments passed to FrameProcessor
        """
        super().__init__(**kwargs)

        self.agent = agent
        self.tool_handler = tool_handler
        self.audit_log = audit_log
        self.llm_client = llm_client
        self.output_dir = output_dir

        # Create agentic system
        self.agentic_system = AgenticSystem(
            current_date_time=current_date_time,
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            llm_client=llm_client,
            output_dir=output_dir,
        )

        # State tracking
        self._aggregator_flush_task = None
        self._user_message_aggregator = []
        self._user_speaking = False
        self._bot_speaking = False

        # Interruption handling
        self._current_query_task: asyncio.Task | None = None
        self._interrupted = asyncio.Event()

        # Optional callback for assistant responses (used for transcript saving)
        self.on_assistant_response: Awaitable | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames and send appropriate messages to the agentic system."""
        # Handle EndFrame and CancelFrame to cleanup
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Check if frame type requires special processing
        if any(isinstance(frame, frame_type) for frame_type in WEBSOCKET_FRAME_TYPES):
            if isinstance(frame, TranscriptionFrame):
                # Log transcription but don't process it
                # Processing happens via on_user_turn_stopped event handler
                logger.info(f"TranscriptionFrame (partial): {frame.text}")
                # Frame will be aggregated by Pipecat's turn management
                # and processed when on_user_turn_stopped fires

            elif isinstance(frame, UserStartedSpeakingFrame):
                logger.info("User started speaking")
                self._user_speaking = True

            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.info("User stopped speaking")
                self._user_speaking = False
                # Processing happens via on_user_turn_stopped event handler

            elif isinstance(frame, BotStartedSpeakingFrame):
                logger.info("Bot started speaking")
                self._bot_speaking = True

            elif isinstance(frame, BotStoppedSpeakingFrame):
                logger.info("Bot stopped speaking")
                self._bot_speaking = False

            elif isinstance(frame, TurnTimestampFrame):
                if frame.role == "assistant":
                    logger.info("Assistant turn ended")

            elif isinstance(frame, SpokenMessageFrame):
                # Log the spoken message
                logger.info(f"Spoken message: {frame.text}")

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def _start_interruption(self):
        """Handle pipecat interruption by cancelling ongoing query processing."""
        self._interrupted.set()
        if self._current_query_task and not self._current_query_task.done():
            logger.info("Interruption received - cancelling ongoing query")
            self._current_query_task.cancel()
            try:
                await self._current_query_task
            except asyncio.CancelledError:
                pass
            self._current_query_task = None
        await super()._start_interruption()

    async def process_complete_user_turn(self, text: str) -> None:
        """Process a complete user turn from Pipecat's turn management.

        This is called by the on_user_turn_stopped event handler with the
        complete user transcript.

        Args:
            text: Complete user message from the turn
        """
        if not text or not text.strip():
            logger.debug("Ignoring empty user turn")
            return

        # Cancel any previous query still running
        if self._current_query_task and not self._current_query_task.done():
            self._current_query_task.cancel()
            try:
                await self._current_query_task
            except asyncio.CancelledError:
                pass

        self._interrupted.clear()
        logger.info(f"Processing complete user turn: {text}")

        # Add to message aggregator for context
        self._user_message_aggregator.append({"role": "user", "content": text})

        # Process through agentic system as a cancellable task
        self._current_query_task = asyncio.create_task(self._process_user_query(text))
        try:
            await self._current_query_task
        except asyncio.CancelledError:
            logger.info("Query processing interrupted by user")
        finally:
            self._current_query_task = None

    async def _process_user_query(self, text: str) -> None:
        """Process a user query through the agentic system."""
        try:
            async for response in self.agentic_system.process_query(text):
                if self._interrupted.is_set():
                    logger.info("Skipping response - interrupted")
                    return
                if response:
                    await self._handle_response(response)
        except asyncio.CancelledError:
            # Pipeline is shutting down - this is expected, just log and exit gracefully
            logger.debug("Query processing cancelled during pipeline shutdown")
            raise  # Re-raise to ensure proper cancellation propagation
        except Exception as e:
            logger.error(f"Error processing user query: {e}", exc_info=True)
            # Try to send error message, but don't fail if pipeline is already closed
            try:
                await self._handle_response("I'm sorry, I encountered an error. Please try again.")
            except Exception:
                logger.debug("Failed to send error message (pipeline may be closed)")

    async def _handle_response(self, message: str) -> None:
        """Handle pushing message to TTS and log it."""
        if self._interrupted.is_set():
            logger.info(f"Skipping speak frame (interrupted): {message}")
            return
        logger.info(f"Pushing speak frame: {message}")

        try:
            # Notify callback for transcript saving
            if self.on_assistant_response:
                await self.on_assistant_response(message)

            # Push content as LLMMessageFrame so that it gets logged as llm_response in pipecat logs
            await self.push_frame(
                LLMMessageFrame(text=message),
                FrameDirection.DOWNSTREAM,
            )
            if len(message) > 1000:
                # chunk into sentences
                sentences = message.split(". ")
                for sentence in sentences:
                    await self.push_frame(
                        TTSSpeakFrame(text=sentence),
                        FrameDirection.DOWNSTREAM,
                    )
            else:
                # Push content to TTS
                await self.push_frame(
                    TTSSpeakFrame(text=message),
                    FrameDirection.DOWNSTREAM,
                )
        except (asyncio.CancelledError, Exception) as e:
            # Pipeline may be cancelled or closed - log at debug level and continue
            logger.debug(f"Failed to push response frame (pipeline may be closed): {e}")
            # Re-raise CancelledError to ensure proper cancellation propagation
            if isinstance(e, asyncio.CancelledError):
                raise

    async def stop(self):
        """Stop the processor and cleanup."""
        logger.info("Stopping AgentProcessor...")

        # Cancel any in-progress query
        self._interrupted.set()
        if self._current_query_task and not self._current_query_task.done():
            self._current_query_task.cancel()
            try:
                await self._current_query_task
            except asyncio.CancelledError:
                pass
            self._current_query_task = None

        # Save agent performance stats
        try:
            logger.info("Calling save_agent_perf_stats()...")
            self.agentic_system.save_agent_perf_stats()
            logger.info("save_agent_perf_stats() completed")
        except Exception as e:
            logger.error(f"Error saving agent performance stats: {e}", exc_info=True)

        # Cancel aggregator flush task if it exists
        if self._aggregator_flush_task:
            logger.debug("Cancelling aggregator flush task on stop")
            try:
                await self._cancel_aggregator_flush()
            except Exception as e:
                logger.debug(f"Error cancelling aggregator flush task during stop: {e}")
            finally:
                self._aggregator_flush_task = None

    async def _cancel_aggregator_flush(self):
        """Cancel any existing aggregator flush task."""
        if self._aggregator_flush_task:
            try:
                await self.cancel_task(self._aggregator_flush_task)
            except Exception:
                pass
            finally:
                self._aggregator_flush_task = None


class UserObserver(FrameProcessor):
    """Observes STT transcription frames and VAD events to track latency metrics and emit user messages."""

    def __init__(self, **kwargs):
        """Initialize the user observer."""
        super().__init__(**kwargs)
        self._last_transcription_time: float | None = None
        self._user_context = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> Awaitable[Frame]:
        """Process frames to track transcription timing and VAD events.

        - Tracks the timing between STT responses and VAD firing to calculate the "VAD buffer" -
          how much latency exists between the last transcription chunk and when speech detection ends.
        - Emits UserMessageFrame for each TranscriptionFrame to allow other processors to observe
          user transcriptions without interfering with the context aggregator.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            # Log interim transcription frames for debugging
            logger.debug(f"Interim transcription received: '{frame.text}'")

        elif isinstance(frame, TranscriptionFrame):
            transcription_text = frame.text.strip()
            if transcription_text:
                # Track final transcription time for VAD buffer calculation
                self._last_transcription_time = time.time()
                self._user_context.append({"role": "user", "content": transcription_text})

                # Log partial transcription (buffered by user_aggregator, not processed immediately)
                logger.info(f"TranscriptionFrame (buffered): '{transcription_text}'")

                # Push UserMessageFrame for metrics/logging (not consumed by aggregator)
                user_message_frame = UserMessageFrame(text=transcription_text)
                await self.push_frame(user_message_frame)

                # NOTE: Do NOT push UserContextFrame here - it triggers immediate LLM processing!
                # The on_user_turn_stopped event handler will process the complete turn.

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Track VAD events (when user stops speaking)
            if self._last_transcription_time is not None:
                current_time = time.time()
                vad_buffer_seconds = current_time - self._last_transcription_time
                vad_buffer_ms = int(vad_buffer_seconds * 1000)

                # Log the VAD buffer metric
                logger.info(f"VAD buffer: {vad_buffer_ms}ms")

                # Push VADBufferFrame for this user turn
                vad_buffer_frame = VADBufferFrame(vad_buffer_ms=vad_buffer_ms)
                await self.push_frame(vad_buffer_frame)

                # Reset for next turn
                self._last_transcription_time = None
            else:
                logger.warning("VAD fired but no previous transcription timestamp found")

        await self.push_frame(frame, direction)


class UserAudioCollector(FrameProcessor):
    """Collects audio frames in a buffer, then adds them to the LLM context when the user stops speaking.

    Used when STT is disabled and we want to pass audio directly to the LLM.
    Based on UserAudioCollector from the original implementation.
    """

    def __init__(self, context, user_context_aggregator):
        """Initialize the audio collector.

        Args:
            context: The OpenAI LLM context
            user_context_aggregator: The user context aggregator
        """
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = VAD_START_SECS
        self._user_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process the frame to collect audio for LLM context."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            # Add collected audio frames to context
            if self._audio_frames:
                await self._context.add_audio_frames_message(audio_frames=self._audio_frames)
                await self._user_context_aggregator.push_frame(LLMContextFrame(context=self._context))

        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Append the audio frame to our buffer. Treat the buffer as a ring buffer,
                # dropping the oldest frames as necessary. Assume all audio frames have the same duration.
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)
