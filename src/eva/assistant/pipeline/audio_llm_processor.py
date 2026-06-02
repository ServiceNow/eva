"""Audio-LLM pipeline processors for self-hosted model (via vLLM).

Provides FrameProcessors for the audio-LLM pipeline:

1. AudioLLMUserAudioCollector: Buffers raw audio frames during user speech.
   Placed before user_aggregator in the pipeline.

2. AudioLLMProcessor: Processes complete user turns via the audio-LLM model.
   Placed after user_aggregator, before TTS. The audio-LLM equivalent of
   BenchmarkAgentProcessor.

3. AudioTranscriptionProcessor: Transcribes audio using chat completions with
   audio input. Can run in parallel with AudioLLMProcessor.

4. InputTranscriptionContextFilter: Helper
   processors for the parallel transcription pipeline.
"""

import asyncio
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    SystemFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

from eva.assistant.agentic.audio_llm_system import AudioLLMAgenticSystem
from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.pipeline.alm_base import DEFAULT_TRANSCRIPTION_PROMPT, BaseALMClient
from eva.assistant.pipeline.frames import LLMMessageFrame
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Pipeline sample rate (matches pipecat_server.py SAMPLE_RATE)
PIPELINE_SAMPLE_RATE = 24000

# Minimum audio size to process (< 10ms of 24kHz 16-bit mono is noise/empty)
MIN_AUDIO_BYTES = 320


class AudioLLMUserAudioCollector(FrameProcessor):
    """Buffers raw audio frames during user speech for the audio-LLM pipeline.

    Audio capture is driven by raw VAD events (UserStartedSpeakingFrame /
    UserStoppedSpeakingFrame), but LLM invocation is gated by the
    smart-turn-aware user_aggregator. Callers invoke ``notify_turn_ended()``
    from the aggregator's ``on_user_turn_stopped`` event (after the turn-stop
    strategy is satisfied), which increments the turn id and pushes an
    LLMContextFrame to trigger both branches of the parallel pipeline.

    Between the first speech segment of a turn and the smart-turn verdict,
    the collector keeps appending audio across VAD start/stop blips so that
    natural pauses (e.g., between phonetic-alphabet characters) don't fragment
    the captured buffer.

    A ring buffer of pre-VAD audio captures the start of speech that occurs
    before VAD fires UserStartedSpeakingFrame.

    All frames pass through unchanged.
    """

    # Default pre-speech buffer (can be overridden via constructor)
    DEFAULT_PRE_SPEECH_SECS = 1.0

    def __init__(
        self,
        context,
        user_context_aggregator,
        pre_speech_secs: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_buffer = bytearray()
        self._pre_speech_buffer: list[bytes] = []
        self._user_speaking = False
        # True between turn boundaries; reset by notify_turn_ended() so the
        # next UserStartedSpeakingFrame starts a fresh buffer instead of
        # appending to the previous turn.
        self._turn_finalized = True
        self._current_turn_id = 0  # Incremented on each finalized turn
        # Pre-speech buffer size (captures audio before VAD fires to avoid cutting off speech)
        self._pre_speech_secs = pre_speech_secs or self.DEFAULT_PRE_SPEECH_SECS
        # Actual sample rate observed on InputAudioRawFrames. Falls back to
        # PIPELINE_SAMPLE_RATE until the first audio frame arrives.
        self._frame_sample_rate: int = PIPELINE_SAMPLE_RATE

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            pre_speech_bytes = b"".join(self._pre_speech_buffer)
            pre_speech_duration_ms = len(pre_speech_bytes) / (self._frame_sample_rate * 2) * 1000
            if self._turn_finalized:
                # Fresh turn: pre-speech becomes the start of the new buffer
                logger.info(
                    f"New turn — prepending {len(pre_speech_bytes)} bytes "
                    f"({pre_speech_duration_ms:.0f}ms) of pre-speech audio "
                    f"(target {self._pre_speech_secs * 1000:.0f}ms, "
                    f"ring chunks={len(self._pre_speech_buffer)})"
                )
                self._audio_buffer = bytearray(pre_speech_bytes)
                self._turn_finalized = False
            else:
                # VAD blip mid-turn (e.g., pause between phonetic letters):
                # append pre-speech so the buffer stays continuous.
                logger.info(
                    f"Resuming turn — appending {len(pre_speech_bytes)} bytes "
                    f"({pre_speech_duration_ms:.0f}ms) of pre-speech audio "
                    f"(target {self._pre_speech_secs * 1000:.0f}ms, "
                    f"ring chunks={len(self._pre_speech_buffer)})"
                )
                self._audio_buffer.extend(pre_speech_bytes)
            self._pre_speech_buffer.clear()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Stop buffering active speech, but DO NOT trigger the LLM here —
            # smart-turn-aware on_user_turn_stopped drives that via
            # notify_turn_ended().
            self._user_speaking = False
            logger.info(
                f"VAD stop — switching to pre-speech buffering "
                f"(audio_buffer_size={len(self._audio_buffer)} bytes, turn_id={self._current_turn_id})"
            )

        elif isinstance(frame, InputAudioRawFrame):
            self._frame_sample_rate = frame.sample_rate
            if self._user_speaking:
                self._audio_buffer.extend(frame.audio)
            else:
                # Ring buffer: keep a rolling window of pre-speech audio
                self._pre_speech_buffer.append(frame.audio)
                # 16-bit mono → 2 bytes per sample
                max_bytes = int(self._pre_speech_secs * self._frame_sample_rate * 2)
                total = sum(len(chunk) for chunk in self._pre_speech_buffer)
                while total > max_bytes and self._pre_speech_buffer:
                    total -= len(self._pre_speech_buffer.pop(0))

        await self.push_frame(frame, direction)

    async def notify_turn_ended(self) -> None:
        """Finalize the current turn and trigger downstream processing.

        Called from pipecat_server's ``on_user_turn_stopped`` handler after
        the smart-turn-aware turn-stop strategy is satisfied. Increments the
        turn id and pushes LLMContextFrame so the transcription branch and
        the audio-LLM branch both observe a consistent turn boundary.
        """
        self._current_turn_id += 1
        self._turn_finalized = True
        await self._user_context_aggregator.push_frame(LLMContextFrame(context=self._context))

    def get_buffered_audio(self) -> bytes:
        """Get the buffered audio and clear the buffer."""
        audio = bytes(self._audio_buffer)
        self._audio_buffer = bytearray()
        return audio

    def peek_buffered_audio(self) -> bytes:
        """Get the buffered audio without clearing the buffer.

        Use this for parallel processing where multiple consumers need the audio.
        """
        return bytes(self._audio_buffer)

    @property
    def has_audio(self) -> bool:
        return len(self._audio_buffer) > 0

    @property
    def current_turn_id(self) -> int:
        """Get the current turn ID for associating transcriptions with entries."""
        return self._current_turn_id

    @property
    def frame_sample_rate(self) -> int:
        """Sample rate observed on the most recent InputAudioRawFrame.

        The audio buffer holds raw PCM at whatever rate the transport delivered
        it; downstream consumers should use this rather than guessing.
        """
        return self._frame_sample_rate


class AudioLLMProcessor(FrameProcessor):
    """Processes complete user turns using the audio-LLM model.

    Placed after user_aggregator in the pipeline. When a user turn ends
    (signaled by on_user_turn_stopped event), this processor:

    1. Gets buffered audio from AudioLLMUserAudioCollector
    2. Sends audio + conversation context to the model for a response
    3. Pushes response text as TTSSpeakFrame for TTS

    No separate transcription is performed — the model receives the raw audio
    directly. A placeholder ``[user audio]`` is used in logs and conversation history.

    This is the Audio-LLM equivalent of BenchmarkAgentProcessor.
    """

    def __init__(
        self,
        current_date_time: str,
        agent: AgentConfig,
        tool_handler: ToolExecutor,
        audit_log: AuditLog,
        alm_client: BaseALMClient,
        audio_collector: AudioLLMUserAudioCollector,
        output_dir: Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.audio_collector = audio_collector
        self.audit_log = audit_log

        # Create agentic system (mirrors BenchmarkAgentProcessor)
        self.agentic_system = AudioLLMAgenticSystem(
            current_date_time=current_date_time,
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            alm_client=alm_client,
            output_dir=output_dir,
        )

        # State tracking (mirrors BenchmarkAgentProcessor)
        self._current_query_task: asyncio.Task | None = None
        self._interrupted = asyncio.Event()

        # Optional callback for transcript saving (set by pipecat_server.py)
        self.on_assistant_response: Awaitable | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        await super().process_frame(frame, direction)

        # Trigger processing when LLMContextFrame is received (from parallel pipeline)
        if isinstance(frame, LLMContextFrame):
            await self.process_complete_user_turn("")
            return

        await self.push_frame(frame, direction)

    async def _start_interruption(self):
        """Handle pipecat interruption by cancelling ongoing query processing."""
        self._interrupted.set()
        if self._current_query_task and not self._current_query_task.done():
            logger.info("Interruption received - cancelling ongoing audio-LLM query")
            self._current_query_task.cancel()
            try:
                await self._current_query_task
            except asyncio.CancelledError:
                pass
            self._current_query_task = None
        await super()._start_interruption()

    async def process_complete_user_turn(self, text_from_aggregator: str) -> None:
        """Process a complete user turn with audio.

        Called by the on_user_turn_stopped event handler in pipecat_server.py.
        The text_from_aggregator is typically empty since there is no STT;

        Args:
            text_from_aggregator: Text from pipecat's turn management
                                  (empty when no STT is configured).
        """
        # Use peek (non-destructive) since transcription processor also needs the audio
        # Buffer is cleared on next UserStartedSpeakingFrame
        audio_bytes = self.audio_collector.peek_buffered_audio()

        if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
            logger.debug("Ignoring user turn with no/tiny audio")
            return

        # Cancel any previous query still running
        if self._current_query_task and not self._current_query_task.done():
            self._current_query_task.cancel()
            try:
                await self._current_query_task
            except asyncio.CancelledError:
                pass

        self._interrupted.clear()
        logger.info(f"Processing audio-LLM user turn ({len(audio_bytes)} bytes)")

        # Add placeholder to audit log BEFORE async task starts.
        # This ensures transcription callback can find and update the entry.
        # Pass turn_id so transcription can update the correct entry.
        turn_id = self.audio_collector.current_turn_id
        self.audit_log.append_user_input(self._USER_PLACEHOLDER, turn_id=turn_id)

        source_sample_rate = self.audio_collector.frame_sample_rate
        self._current_query_task = asyncio.create_task(self._process_audio_turn(audio_bytes, source_sample_rate))
        try:
            await self._current_query_task
        except asyncio.CancelledError:
            logger.info("Audio-LLM query processing interrupted by user")
        finally:
            self._current_query_task = None

    # Placeholder used in audit_log / transcript since no real transcription is available
    _USER_PLACEHOLDER = "[user audio]"

    async def _process_audio_turn(self, audio_bytes: bytes, source_sample_rate: int) -> None:
        """Process a user turn with audio data."""
        try:
            # Send audio to the agentic system and process
            self.agentic_system.set_turn_audio(audio_bytes, source_sample_rate)

            async for response in self.agentic_system.process_query_with_audio(self._USER_PLACEHOLDER):
                if self._interrupted.is_set():
                    logger.info("Skipping response - interrupted")
                    return
                if response:
                    await self._handle_response(response)

        except asyncio.CancelledError:
            logger.debug("Audio turn processing cancelled during pipeline shutdown")
            raise
        except Exception as e:
            logger.error(f"Error processing audio turn: {e}", exc_info=True)
            try:
                await self._handle_response("I'm sorry, I encountered an error. Please try again.")
            except Exception:
                logger.debug("Failed to send error message (pipeline may be closed)")

    async def _handle_response(self, message: str) -> None:
        """Push response to TTS. Mirrors BenchmarkAgentProcessor._handle_response."""
        if self._interrupted.is_set():
            logger.info(f"Skipping speak frame (interrupted): {message}")
            return
        logger.info(f"Pushing speak frame: {message}")

        try:
            # Notify callback for transcript saving
            if self.on_assistant_response:
                await self.on_assistant_response(message)

            # Push content as LLMMessageFrame for pipecat log observers
            await self.push_frame(LLMMessageFrame(text=message), FrameDirection.DOWNSTREAM)

            if len(message) > 1000:
                # Chunk long messages into sentences for better TTS streaming
                sentences = message.split(". ")
                for sentence in sentences:
                    await self.push_frame(TTSSpeakFrame(text=sentence), FrameDirection.DOWNSTREAM)
            else:
                await self.push_frame(TTSSpeakFrame(text=message), FrameDirection.DOWNSTREAM)
        except (asyncio.CancelledError, Exception) as e:
            logger.debug(f"Failed to push response frame (pipeline may be closed): {e}")
            if isinstance(e, asyncio.CancelledError):
                raise

    async def stop(self):
        """Stop the processor and cleanup."""
        logger.info("Stopping AudioLLMProcessor...")

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
            logger.info("Saving audio-LLM agent perf stats...")
            self.agentic_system.save_agent_perf_stats()
        except Exception as e:
            logger.error(f"Error saving agent perf stats: {e}", exc_info=True)


# =============================================================================
# Parallel Transcription Pipeline Processors
# =============================================================================


class InputTranscriptionContextFilter(FrameProcessor):
    """Filters frames for the transcription branch of the parallel pipeline.

    Blocks all frames except LLMContextFrame (and SystemFrame for lifecycle).
    Extracts audio from the context and passes it to the next processor.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            # Pass through system frames for pipeline lifecycle
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, LLMContextFrame):
            return

        # Pass through the LLMContextFrame - the transcription processor will handle it
        await self.push_frame(frame, direction)


class AudioTranscriptionProcessor(FrameProcessor):
    """Transcribes audio using chat completions with audio input.

    Gets audio from the AudioLLMUserAudioCollector and sends it to an audio-capable
    LLM (like gpt-4o-audio-preview) via chat completions, emitting the transcription
    as an LLMDemoTranscriptionFrame.

    This is more generic than the dedicated transcription API and allows:
    - Custom system prompts for transcription
    - Using any audio-capable chat model
    - Additional context or instructions

    This processor can be triggered by LLMContextFrame events from the parallel
    pipeline, or called directly via the `transcribe()` method from event handlers.

    Set `on_transcription` callback to receive transcription text for logging.

    Transcription runs as a background task so it can complete even if the user
    starts speaking again (interruption). This ensures transcriptions are not lost.
    """

    def __init__(
        self,
        audio_collector: AudioLLMUserAudioCollector,
        alm_client: BaseALMClient,
        system_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._audio_collector = audio_collector
        self._alm_client = alm_client
        self._system_prompt = system_prompt or DEFAULT_TRANSCRIPTION_PROMPT

        # Callback for when transcription is ready (set by pipecat_server.py)
        self.on_transcription: Any | None = None

        # Track background transcription tasks so they can complete even during interruptions
        self._transcription_tasks: list[asyncio.Task] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames from the pipeline (kept for compatibility)."""
        # Handle pipeline shutdown - wait for pending transcriptions
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.cleanup()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        # Capture turn_id, audio, and sample rate at the moment we receive the frame
        turn_id = self._audio_collector.current_turn_id
        # Capture audio NOW before it gets overwritten by the next turn
        audio_data = self._audio_collector.peek_buffered_audio()
        source_sample_rate = self._audio_collector.frame_sample_rate
        logger.info(f"transcribe (turn_id={turn_id})")
        timestamp = time_now_iso8601()

        # Run transcription as background task so it completes even if interrupted
        task = asyncio.create_task(
            self._transcribe_audio(audio_data, timestamp, source_sample_rate, turn_id=turn_id)
        )
        self._transcription_tasks.append(task)
        # Clean up completed tasks
        self._transcription_tasks = [t for t in self._transcription_tasks if not t.done()]

    async def transcribe(self, timestamp: str, turn_id: int | None = None) -> str | None:
        """Transcribe audio from the collector using chat completions.

        This method can be called directly from event handlers or via frame processing.
        For internal use, prefer _transcribe_audio() with pre-captured audio data.

        Args:
            timestamp: ISO8601 timestamp for the transcription.
            turn_id: Optional turn identifier for associating with audit log entry.

        Returns:
            The transcription text, or None if transcription failed or audio was empty.
        """
        audio_data = self._audio_collector.peek_buffered_audio()
        source_sample_rate = self._audio_collector.frame_sample_rate
        return await self._transcribe_audio(audio_data, timestamp, source_sample_rate, turn_id)

    async def _transcribe_audio(
        self,
        audio_data: bytes,
        timestamp: str,
        source_sample_rate: int,
        turn_id: int | None = None,
    ) -> str | None:
        """Transcribe pre-captured audio data using chat completions.

        This method takes audio data directly instead of reading from the collector,
        ensuring the correct audio is transcribed even if the buffer is overwritten
        by a subsequent turn.

        Args:
            audio_data: Raw PCM audio bytes to transcribe.
            timestamp: ISO8601 timestamp for the transcription.
            turn_id: Optional turn identifier for associating with audit log entry.

        Returns:
            The transcription text, or None if transcription failed or audio was empty.
        """
        try:
            if not audio_data or len(audio_data) < MIN_AUDIO_BYTES:
                logger.info("No/insufficient audio data for transcription")
                return None

            start_time = time.time()
            text = await self._alm_client.transcribe(
                audio_bytes=audio_data,
                source_sample_rate=source_sample_rate,
                system_prompt=self._system_prompt,
            )
            elapsed = time.time() - start_time

            if not text:
                logger.info(f"Empty transcription (turn_id={turn_id})")
                return None

            logger.info(f"Transcription from {self._alm_client.model}: {text}")
            logger.debug(f"Elapsed time: {elapsed:.2f}s")

            if self.on_transcription and text != "EMPTY":
                await self.on_transcription(text, timestamp, turn_id)

            return text

        except asyncio.CancelledError:
            logger.warning(f"Transcription cancelled for turn_id={turn_id}")
            raise
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)
            return None

    async def cleanup(self) -> None:
        """Wait for pending transcription tasks to complete."""
        if self._transcription_tasks:
            pending = [t for t in self._transcription_tasks if not t.done()]
            if pending:
                logger.info(f"Waiting for {len(pending)} pending transcription(s) to complete...")
                await asyncio.gather(*pending, return_exceptions=True)
            self._transcription_tasks.clear()
