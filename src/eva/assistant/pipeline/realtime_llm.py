"""Instrumented realtime LLM service for correct audit log ordering and timestamps.

Subclasses OpenAIRealtimeLLMService to intercept raw OpenAI Realtime API events
(speech_started, speech_stopped, transcription.completed, response.done) which
have a guaranteed ordering and carry item_id for correlation.

The OpenAI Realtime API fires events in this order per turn:
1. speech_started -> 2. speech_stopped -> 3. transcription.completed
4. audio_delta / audio_transcript_delta -> 5. response.done

Writing user entries on #3 and assistant entries on #5 guarantees correct order.
"""

import struct
import time
from dataclasses import dataclass
from typing import Any, Optional

from pipecat.frames.frames import Frame, InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from eva.assistant.agentic.audit_log import AuditLog
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Audio threshold for detecting speech vs silence
# RMS values below this are considered silence
SILENCE_RMS_THRESHOLD = 10


@dataclass
class _UserTurnRecord:
    """Tracks state for a single user speech turn identified by item_id."""

    item_id: str
    speech_started_wall_ms: str = ""
    speech_stopped_wall_ms: str = ""
    transcript: str = ""
    flushed: bool = False


def _wall_ms() -> str:
    """Return current wall-clock time as epoch milliseconds string."""
    return str(int(round(time.time() * 1000)))


def _calculate_rms(audio_bytes: bytes) -> float:
    """Calculate RMS (root mean square) energy of 16-bit PCM audio."""
    if len(audio_bytes) < 2:
        return 0.0
    num_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", audio_bytes[: num_samples * 2])
    if not samples:
        return 0.0
    sum_squares = sum(s * s for s in samples)
    return (sum_squares / len(samples)) ** 0.5


class InstrumentedRealtimeLLMService(OpenAIRealtimeLLMService):
    """OpenAIRealtimeLLMService subclass that writes audit log entries with correct ordering and wall-clock timestamps derived from Realtime API events.

    All overridden methods call ``super()`` first so that the parent's frame
    processing (audio playback, interruption handling, metrics, etc.) is fully
    preserved.
    """

    def __init__(self, *, audit_log: AuditLog, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audit_log = audit_log

        # Per-user-turn state, keyed by item_id
        self._user_turns: dict[str, _UserTurnRecord] = {}

        # Assistant response accumulation (across audio_transcript_delta events)
        self._current_assistant_transcript_parts: list[str] = []
        self._assistant_response_start_wall_ms: Optional[str] = None

        # Track whether we're mid-assistant-response (for interruption flushing)
        self._assistant_responding: bool = False

        # Track audio frame timing for VAD delay calculation
        self._last_audio_frame_time: Optional[float] = None
        self._vad_delay_ms: Optional[int] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Track audio frame timing before passing to parent.

        Only updates the timestamp when audio has actual speech content (not silence),
        so VAD delay calculation reflects when user actually stopped speaking.
        """
        if isinstance(frame, InputAudioRawFrame):
            rms = _calculate_rms(frame.audio)
            if rms > SILENCE_RMS_THRESHOLD:
                self._last_audio_frame_time = time.time()

        await super().process_frame(frame, direction)

    async def _handle_evt_speech_started(self, evt: Any) -> None:
        """Fires when user starts speaking (input_audio_buffer.speech_started).

        Captures wall-clock start time.  Also flushes any in-progress
        interrupted assistant response before recording the new user turn.
        """
        # Reset VAD tracking for new turn
        self._vad_delay_ms = None

        # Broadcast VAD user started speaking frame because realtime VAD does not broadcast it themselves
        await self.broadcast_frame(VADUserStartedSpeakingFrame)

        # Flush interrupted assistant response if one is in progress
        if self._assistant_responding and self._current_assistant_transcript_parts:
            partial_text = "".join(self._current_assistant_transcript_parts) + " [interrupted]"
            self._audit_log.append_assistant_output(
                partial_text,
                timestamp_ms=self._assistant_response_start_wall_ms,
            )
            logger.debug(f"Flushed interrupted assistant response: {partial_text[:60]}...")
            self._current_assistant_transcript_parts.clear()
            self._assistant_response_start_wall_ms = None
            self._assistant_responding = False

        # Let parent handle frame processing (interruption, UserStartedSpeaking, etc.)
        await super()._handle_evt_speech_started(evt)

        item_id = getattr(evt, "item_id", None) or ""
        wall = _wall_ms()

        record = _UserTurnRecord(item_id=item_id, speech_started_wall_ms=wall)
        self._user_turns[item_id] = record
        logger.debug(f"speech_started: item_id={item_id}, wall_ms={wall}")

    async def _handle_evt_speech_stopped(self, evt: Any) -> None:
        """Fires when user stops speaking (input_audio_buffer.speech_stopped).

        Captures wall-clock end time for the user turn and calculates VAD delay.
        """
        speech_stopped_time = time.time()

        # Calculate VAD delay: time between last audio frame and speech_stopped event
        if self._last_audio_frame_time is not None:
            self._vad_delay_ms = int((speech_stopped_time - self._last_audio_frame_time) * 1000)
        else:
            logger.warning("speech_stopped fired but no audio frames were tracked")
            self._vad_delay_ms = None

        # Reset audio tracking for next turn
        self._last_audio_frame_time = None

        await self.broadcast_frame(VADUserStoppedSpeakingFrame)
        await super()._handle_evt_speech_stopped(evt)

        item_id = getattr(evt, "item_id", None) or ""
        wall = _wall_ms()

        record = self._user_turns.get(item_id)
        if record:
            record.speech_stopped_wall_ms = wall
        else:
            # speech_stopped without prior speech_started — create a record
            record = _UserTurnRecord(
                item_id=item_id,
                speech_stopped_wall_ms=wall,
            )
            self._user_turns[item_id] = record

        logger.debug(f"speech_stopped: item_id={item_id}, wall_ms={wall}")

    async def handle_evt_input_audio_transcription_completed(self, evt: Any) -> None:
        """Fires when input audio transcription completes.

        Writes the user entry to the audit log using the speech_started wall
        clock as the timestamp (so it reflects when the user actually spoke).
        """
        await super().handle_evt_input_audio_transcription_completed(evt)

        item_id = getattr(evt, "item_id", None) or ""
        transcript = getattr(evt, "transcript", None) or ""

        if not transcript or not transcript.strip():
            logger.debug(f"transcription_completed: empty transcript for item_id={item_id}, skipping")
            return

        record = self._user_turns.get(item_id)
        if not record:
            logger.warning(f"transcription_completed: no speech_started record for item_id={item_id}, skipping")
            return

        self._audit_log.append_user_input(transcript.strip(), timestamp_ms=record.speech_started_wall_ms)
        logger.debug(
            f"transcription_completed: item_id={item_id}, transcript='{transcript[:50]}...', timestamp_ms={record.speech_started_wall_ms}"
        )

        # Mark as flushed
        record.transcript = transcript
        record.flushed = True

    async def _handle_evt_audio_delta(self, evt: Any) -> None:
        """Fires for each audio chunk of the assistant response.

        Captures wall-clock of the *first* delta as assistant response start.
        Also logs the full user-perceived response latency including VAD delay.
        """
        await super()._handle_evt_audio_delta(evt)

        if self._assistant_response_start_wall_ms is None:
            self._assistant_response_start_wall_ms = _wall_ms()
            self._assistant_responding = True

            # Log full user-perceived latency (includes VAD delay)
            if self._vad_delay_ms is not None:
                # Find the most recent user turn to get speech_stopped time
                recent_record = None
                for record in self._user_turns.values():
                    if record.speech_stopped_wall_ms:
                        recent_record = record

                if recent_record and recent_record.speech_stopped_wall_ms:
                    speech_stopped_ms = int(recent_record.speech_stopped_wall_ms)
                    response_start_ms = int(self._assistant_response_start_wall_ms)
                    vad_to_response_ms = response_start_ms - speech_stopped_ms
                    full_latency_ms = vad_to_response_ms + self._vad_delay_ms
                    logger.debug(
                        f"Full response latency: {full_latency_ms}ms "
                        f"(VAD delay: {self._vad_delay_ms}ms + response: {vad_to_response_ms}ms)"
                    )

    async def _handle_evt_audio_transcript_delta(self, evt: Any) -> None:
        """Fires for incremental assistant transcript text.

        Accumulates transcript parts for the current assistant response.
        """
        await super()._handle_evt_audio_transcript_delta(evt)

        delta = getattr(evt, "delta", None) or ""
        if delta:
            self._current_assistant_transcript_parts.append(delta)

    async def _handle_evt_response_done(self, evt: Any) -> None:
        """Fires when the assistant response is complete (response.done).

        Writes the assistant entry to the audit log.  Extracts text from
        ``evt.response.output`` items, falling back to accumulated transcript
        deltas.

        Responses that produced no audible speech are skipped:
        - Tool-call-only responses (no audio/text output items at all).
        - Mixed responses (text + function_call) where no ``audio_delta`` was
          ever received — the model decided to call a tool before streaming
          any audio, so the text was never spoken.
        """
        await super()._handle_evt_response_done(evt)

        audio_was_streamed = self._assistant_response_start_wall_ms is not None
        has_function_calls = self._response_has_function_calls(evt)

        # Try to extract text from the response output items
        content = self._extract_response_text(evt)

        # Fallback to accumulated transcript deltas
        if not content:
            content = "".join(self._current_assistant_transcript_parts)

        # Decide whether this response produced audible speech worth logging.
        if not content and has_function_calls:
            # Tool-call-only response with no text — nothing was spoken.
            logger.debug("response_done: tool-call-only response, skipping assistant entry")
            self._reset_assistant_state()
            return

        if content and not audio_was_streamed and has_function_calls:
            # Mixed response: the model generated text alongside tool calls
            # but never actually streamed audio — the text was never spoken.
            logger.debug(
                f"response_done: mixed response with no audio streamed, skipping unsaid text: '{content[:60]}...'"
            )
            self._reset_assistant_state()
            return

        if not content:
            # Genuine audio-only response where transcript is unavailable.
            content = "[audio response - transcription unavailable]"

        timestamp = self._assistant_response_start_wall_ms or _wall_ms()
        self._audit_log.append_assistant_output(content, timestamp_ms=timestamp)
        logger.debug(f"response_done: content='{content[:60]}...', timestamp_ms={timestamp}")

        self._reset_assistant_state()

    def _reset_assistant_state(self) -> None:
        """Clear accumulated assistant response state."""
        self._current_assistant_transcript_parts.clear()
        self._assistant_response_start_wall_ms = None
        self._assistant_responding = False

    @property
    def last_vad_delay_ms(self) -> Optional[int]:
        """Return the most recent VAD delay in milliseconds.

        This is the time between when audio frames stopped arriving and when
        OpenAI's VAD detected end of speech. Can be used to adjust response
        latency measurements to reflect user-perceived latency.
        """
        return self._vad_delay_ms

    @staticmethod
    def _response_has_function_calls(evt: Any) -> bool:
        """Return True if the response.done event contains any function_call outputs."""
        response = getattr(evt, "response", None)
        if not response:
            return False
        output_items = getattr(response, "output", None) or []
        return any(getattr(item, "type", "") == "function_call" for item in output_items)

    @staticmethod
    def _extract_response_text(evt: Any) -> str:
        """Extract text content from a response.done event's output items."""
        response = getattr(evt, "response", None)
        if not response:
            return ""

        output_items = getattr(response, "output", None) or []
        text_parts: list[str] = []

        for item in output_items:
            # Each output item may have a `content` list with typed parts
            content_list = getattr(item, "content", None) or []
            for part in content_list:
                part_type = getattr(part, "type", "")
                if part_type == "audio" or part_type == "text":
                    transcript = getattr(part, "transcript", None) or getattr(part, "text", None) or ""
                    if transcript:
                        text_parts.append(transcript)

        return "".join(text_parts).strip()
