"""Pipeline observers for logging and turn tracking."""

import json
import time
from pathlib import Path
from typing import Any

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from eva.assistant.pipeline.frames import LLMMessageFrame, UserContextFrame
from eva.utils.logging import get_logger

logger = get_logger(__name__)


_TRANSCRIPTION_SERVICES = (STTService, AzureRealtimeLLMService, OpenAIRealtimeLLMService)


class WallClock(SystemClock):
    """SystemClock that also records wall clock time at start for timestamp conversion."""

    def __init__(self):
        super().__init__()
        self.start_wall_time_ns: int = 0

    def start(self):
        self.start_wall_time_ns = time.time_ns()
        super().start()

    def to_wall_time(self, elapsed_ns: int) -> int:
        """Convert elapsed nanoseconds to wall clock time (milliseconds since epoch)."""
        return (self.start_wall_time_ns + elapsed_ns) // 1_000_000


class BenchmarkLogObserver(TurnTrackingObserver):
    """Custom log observer that saves context frames and responses to a JSONL file.

    Turn tracking logic:
    - The first turn starts immediately when the pipeline starts (StartFrame)
    - Subsequent turns start when the user starts speaking
    - A turn ends when the bot stops speaking and either:
      - The user starts speaking again
      - A timeout period elapses with no more bot speech
    """

    def __init__(self, output_path: str, conversation_id: str, clock: WallClock, turn_end_timeout_secs: float = 5.0):
        """Initialize the log observer.

        Args:
            output_path: Directory to save the JSONL file
            conversation_id: The conversation ID for the current session
            clock: WallClock instance for converting timestamps to wall clock time
            turn_end_timeout_secs: Timeout in seconds for turn end detection
        """
        super().__init__(turn_end_timeout_secs=turn_end_timeout_secs)
        self.output_path = Path(output_path)
        self.conversation_id = conversation_id
        self.clock = clock
        self.log_file = self.output_path / "pipecat_logs.jsonl"

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    def write_log_entry(self, entry: dict) -> None:
        """Write a log entry to the JSONL file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")

    async def on_push_frame(self, data: FramePushed) -> None:
        """Log relevant frames to the JSONL file."""
        src = data.source
        dst = data.destination
        frame = data.frame
        timestamp = self.clock.to_wall_time(data.timestamp)

        # Call parent for turn tracking
        await super().on_push_frame(data)

        # We omit logs not from AgentProcessor as a source because they are duplicated by subsequent frame processors
        relevant_logs = (
            isinstance(src, LLMService)
            or isinstance(dst, LLMService)
            or str(src).startswith("BenchmarkAgentProcessor")
            or str(src).startswith("AgentProcessor")
        )

        common_args = {
            "timestamp": timestamp,
            "source": str(src),
            "destination": str(dst),
            "conversation_id": self.conversation_id,
        }

        log_entry = None

        if isinstance(frame, LLMTextFrame) and relevant_logs:
            log_entry = {
                "type": "llm_response",
                "data": {"frame": str(frame.text)},
                **common_args,
            }
        elif isinstance(frame, TranscriptionFrame) and isinstance(src, _TRANSCRIPTION_SERVICES):
            log_entry = {
                "type": "transcript",
                "data": {"frame": str(frame.text)},
                **common_args,
            }
        elif isinstance(frame, LLMMessageFrame) and relevant_logs:
            log_entry = {
                "type": "llm_response",
                "data": {"frame": str(frame.text)},
                **common_args,
            }
        elif isinstance(frame, UserContextFrame) and relevant_logs:
            log_entry = {
                "type": "llm_context",
                "data": {"frame": str(frame.messages)},
                **common_args,
            }
        elif isinstance(frame, TTSTextFrame) and isinstance(src, TTSService):
            log_entry = {
                "type": "tts_text",
                "data": {"frame": str(frame.text)},
                **common_args,
            }
        elif (
            isinstance(frame, UserStartedSpeakingFrame)
            and str(src).startswith("LLMUserAggregator")
            and not str(dst).startswith("UserObserver")  # Avoid duplicate logs
        ):
            log_entry = {
                "type": "user_started_speaking",
                "data": {"frame": str(frame)},
                **common_args,
            }
        elif (
            isinstance(frame, UserStoppedSpeakingFrame)
            and str(src).startswith("LLMUserAggregator")
            and not str(dst).startswith("UserObserver")  # Avoid duplicate logs
        ):
            log_entry = {
                "type": "user_stopped_speaking",
                "data": {"frame": str(frame)},
                **common_args,
            }
        elif isinstance(frame, ErrorFrame) and str(src).startswith("BenchmarkAgentProcessor"):
            log_entry = {
                "type": "error",
                "data": {"frame": str(frame)},
                **common_args,
            }

        if log_entry:
            self.write_log_entry(log_entry)

    async def _start_turn(self, data: FramePushed) -> None:
        """Start a new turn."""
        await super()._start_turn(data)
        logger.info(f"Turn started - {self.conversation_id}")

        src = data.source
        dst = data.destination
        frame = data.frame
        timestamp = self.clock.to_wall_time(data.timestamp)

        common_args = {
            "timestamp": timestamp,
            "source": str(src),
            "destination": str(dst),
            "conversation_id": self.conversation_id,
        }
        self.write_log_entry(
            {
                "type": "turn_start",
                "data": {"frame": str(frame)},
                **common_args,
            }
        )

    async def _end_turn(self, data: FramePushed, was_interrupted: bool) -> None:
        """End the current turn."""
        await super()._end_turn(data, was_interrupted)
        logger.info(f"Turn ended - {self.conversation_id} (interrupted: {was_interrupted})")

        src = data.source
        dst = data.destination
        frame = data.frame
        timestamp = self.clock.to_wall_time(data.timestamp)

        common_args = {
            "timestamp": timestamp,
            "source": str(src),
            "destination": str(dst),
            "conversation_id": self.conversation_id,
        }
        self.write_log_entry(
            {
                "type": "turn_end",
                "data": {"frame": str(frame), "was_interrupted": was_interrupted},
                **common_args,
            }
        )


class MetricsFileObserver(BaseObserver):
    """Observer that writes Pipecat MetricsFrame data to a JSONL file.

    This observer intercepts MetricsFrame instances flowing through the pipeline
    and writes them to a JSONL file for later analysis by EVA metrics.
    """

    def __init__(self, output_path: Path | str, clock: WallClock):
        """Initialize the metrics file observer.

        Args:
            output_path: Path to the JSONL file to write metrics to
            clock: WallClock instance for converting timestamps to wall clock time
        """
        super().__init__()
        self.output_path = Path(output_path)
        self.clock = clock
        self._frames_seen: set[int] = set()
        self._file_handle = None

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file for writing
        self._file_handle = open(self.output_path, "w")
        logger.debug(f"MetricsFileObserver writing to {self.output_path}")

    def _serialize_value(self, value: Any) -> Any:
        """Serialize metric value to JSON-compatible format.

        Args:
            value: The metric value to serialize

        Returns:
            JSON-serializable value
        """
        if isinstance(value, LLMTokenUsage):
            return {
                "prompt_tokens": value.prompt_tokens,
                "completion_tokens": value.completion_tokens,
                "total_tokens": value.total_tokens,
                "cache_creation_input_tokens": getattr(value, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(value, "cache_read_input_tokens", None),
            }
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        else:
            return str(value)

    def _write_jsonl(self, entry: dict):
        """Write a single entry to the JSONL file.

        Args:
            entry: Dictionary to write as JSON line
        """
        if self._file_handle:
            self._file_handle.write(json.dumps(entry) + "\n")
            self._file_handle.flush()

    async def on_push_frame(self, data: FramePushed):
        """Handle pushed frames, intercepting MetricsFrame instances.

        Args:
            data: The frame push event data
        """
        frame = data.frame

        # Only process MetricsFrame instances
        if not isinstance(frame, MetricsFrame):
            return

        # Deduplicate based on frame ID
        frame_id = id(frame)
        if frame_id in self._frames_seen:
            return
        self._frames_seen.add(frame_id)

        timestamp = self.clock.to_wall_time(data.timestamp)

        # Process each metric in the frame
        for metrics_data in frame.data:
            metric_type = type(metrics_data).__name__

            # Most MetricsData subclasses have a `value` field, but some
            # (e.g. SmartTurnMetricsData) use domain-specific fields instead.
            if hasattr(metrics_data, "value"):
                serialized_value = self._serialize_value(metrics_data.value)
            else:
                # Serialize all non-base fields for metric types without `value`
                base_fields = {"processor", "model"}
                serialized_value = {
                    k: self._serialize_value(v) for k, v in metrics_data.model_dump().items() if k not in base_fields
                }

            entry = {
                "timestamp": timestamp,
                "type": metric_type,
                "processor": metrics_data.processor,
                "model": metrics_data.model,
                "value": serialized_value,
            }

            self._write_jsonl(entry)

            logger.debug(
                f"Metrics: {metric_type} from {metrics_data.processor} (model={metrics_data.model}): {serialized_value}"
            )

    def close(self) -> None:
        """Flush and close the file handle. Call this explicitly during teardown."""
        if self._file_handle:
            try:
                self._file_handle.flush()
                self._file_handle.close()
            except Exception:
                pass
            finally:
                self._file_handle = None
            logger.debug(f"MetricsFileObserver closed {self.output_path}")

    def __del__(self):
        """Fallback cleanup — prefer calling close() explicitly."""
        self.close()
