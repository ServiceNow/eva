"""Event logger for ElevenLabs conversation events."""

import json
import time
from pathlib import Path
from typing import Any

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class ElevenLabsEventLogger:
    """Logs events from ElevenLabs conversations for analysis.

    Events are stored in JSONL format for easy processing by the metrics system.
    """

    def __init__(self, output_path: Path):
        """Initialize the event logger.

        Args:
            output_path: Path to the output JSONL file
        """
        self.output_path = output_path
        self._events: list[dict[str, Any]] = []
        self._sequence = 0

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event.

        Args:
            event_type: Type of event (e.g., 'user_message', 'assistant_response')
            data: Event data
        """
        self._sequence += 1
        event = {
            "timestamp": int(time.time() * 1000),
            "sequence": self._sequence,
            "type": event_type,
            "data": data,
        }
        self._events.append(event)
        logger.debug(f"ElevenLabs event: {event_type}")

    def log_user_speech(self, text: str, is_final: bool = True) -> None:
        """Log user speech transcription."""
        self.log_event(
            "user_speech",
            {
                "text": text,
                "is_final": is_final,
            },
        )

    def log_assistant_speech(self, text: str) -> None:
        """Log assistant speech."""
        self.log_event(
            "assistant_speech",
            {
                "text": text,
            },
        )

    def log_audio_sent(self, size_bytes: int) -> None:
        """Log audio data sent to assistant."""
        self.log_event(
            "audio_sent",
            {
                "size_bytes": size_bytes,
            },
        )

    def log_audio_received(self, size_bytes: int) -> None:
        """Log audio data received from assistant."""
        self.log_event(
            "audio_received",
            {
                "size_bytes": size_bytes,
            },
        )

    def log_connection_state(self, state: str, details: dict[str, Any] | None = None) -> None:
        """Log connection state change."""
        self.log_event(
            "connection_state",
            {
                "state": state,
                "details": details or {},
            },
        )

    def log_error(self, error: str, details: dict[str, Any] | None = None) -> None:
        """Log an error."""
        self.log_event(
            "error",
            {
                "error": error,
                "details": details or {},
            },
        )

    def log_audio_start(self, role: str) -> None:
        """Log when audio starts for a given role.

        Args:
            role: Either 'elevenlabs_user' or 'pipecat_agent'
        """
        # Use Unix timestamp in seconds (as float)
        audio_timestamp = time.time()
        # Note: For audio events, we need to store event_type and user at top level
        # not nested in data
        self._sequence += 1
        event = {
            "timestamp": int(time.time() * 1000),  # Keep milliseconds for consistency
            "sequence": self._sequence,
            "event_type": "audio_start",
            "user": role,
            "audio_timestamp": audio_timestamp,  # Unix timestamp in seconds for audio timing
        }
        self._events.append(event)
        logger.debug(f"Audio start logged: {role}")

    def log_audio_end(self, role: str) -> None:
        """Log when audio ends for a given role.

        Args:
            role: Either 'elevenlabs_user' or 'pipecat_agent'
        """
        # Use Unix timestamp in seconds (as float)
        audio_timestamp = time.time()
        # Note: For audio events, we need to store event_type and user at top level
        # not nested in data
        self._sequence += 1
        event = {
            "timestamp": int(time.time() * 1000),  # Keep milliseconds for consistency
            "sequence": self._sequence,
            "event_type": "audio_end",
            "user": role,
            "audio_timestamp": audio_timestamp,  # Unix timestamp in seconds for audio timing
        }
        self._events.append(event)
        logger.debug(f"Audio end logged: {role}")

    def save(self) -> None:
        """Save all logged events to the output file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            f.writelines(json.dumps(event) + "\n" for event in self._events)

        logger.info(f"Saved {len(self._events)} ElevenLabs events to {self.output_path}")

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get logged events, optionally filtered by type.

        Args:
            event_type: Optional event type to filter by

        Returns:
            List of events
        """
        if event_type is None:
            return self._events.copy()
        return [e for e in self._events if e["type"] == event_type]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of logged events."""
        event_counts: dict[str, int] = {}
        for event in self._events:
            event_type = event["type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self._events),
            "event_counts": event_counts,
        }

    def clear(self) -> None:
        """Clear all logged events."""
        self._events.clear()
        self._sequence = 0
