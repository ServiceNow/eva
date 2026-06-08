"""Event loggers for provider-neutral and legacy simulator artifacts."""

import json
import time
from pathlib import Path
from typing import Any

from eva.utils.logging import get_logger

logger = get_logger(__name__)


_NEUTRAL_ROLE = {
    "elevenlabs_user": "simulated_user",
    "framework_agent": "assistant",
    "pipecat_agent": "assistant",
}
_LEGACY_ROLE = {"simulated_user": "elevenlabs_user", "assistant": "framework_agent"}
_NEUTRAL_SOURCE = {"elevenlabs_agent": "simulated_user", "pipecat_assistant": "assistant"}
_LEGACY_SOURCE = {"simulated_user": "elevenlabs_agent", "assistant": "pipecat_assistant"}


class UserSimulatorEventLogger:
    """Logs provider-neutral simulator events for metrics processing.

    Events are stored in JSONL format for easy processing by the metrics system.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        provider: str,
        legacy_output_path: Path | None = None,
        normalize_roles: bool = True,
        include_provider: bool = True,
    ):
        """Initialize the event logger.

        Args:
            output_path: Path to the output JSONL file
            provider: Provider identifier stored with neutral events.
            legacy_output_path: Optional path for a legacy ElevenLabs-compatible copy.
            normalize_roles: Whether to convert historical role names to neutral names.
            include_provider: Whether serialized events include the provider identifier.
        """
        self.output_path = output_path
        self.provider = provider
        self.legacy_output_path = legacy_output_path
        self.normalize_roles = normalize_roles
        self.include_provider = include_provider
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
            "data": self._normalize_data(data),
        }
        if self.include_provider:
            event["provider"] = self.provider
        self._events.append(event)
        logger.debug(f"User simulator event: {event_type}")

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

    def log_audio_start(self, role: str, timestamp: float | None = None) -> None:
        """Log when audio starts for a given role.

        Args:
            role: Provider-neutral or legacy speaker role.
            timestamp: Timestamp in milliseconds when audio started
        """
        # Use Unix timestamp in seconds (as float)
        audio_timestamp = timestamp or time.time()
        # Note: For audio events, we need to store event_type and user at top level
        # not nested in data
        self._sequence += 1
        event = {
            "timestamp": int(time.time() * 1000),  # Keep milliseconds for consistency
            "sequence": self._sequence,
            "event_type": "audio_start",
            "user": self._normalize_role(role),
            "audio_timestamp": audio_timestamp,  # Unix timestamp in seconds for audio timing
        }
        if self.include_provider:
            event["provider"] = self.provider
        self._events.append(event)
        logger.debug(f"Audio start logged: {role}")

    def log_audio_end(self, role: str) -> None:
        """Log when audio ends for a given role.

        Args:
            role: Provider-neutral or legacy speaker role.
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
            "user": self._normalize_role(role),
            "audio_timestamp": audio_timestamp,  # Unix timestamp in seconds for audio timing
        }
        if self.include_provider:
            event["provider"] = self.provider
        self._events.append(event)
        logger.debug(f"Audio end logged: {role}")

    def save(self) -> None:
        """Save all logged events to the output file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            f.writelines(json.dumps(event) + "\n" for event in self._events)

        if self.legacy_output_path is not None:
            self.legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.legacy_output_path, "w") as f:
                f.writelines(json.dumps(self._to_legacy_event(event)) + "\n" for event in self._events)

        logger.info(f"Saved {len(self._events)} user simulator events to {self.output_path}")

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get logged events, optionally filtered by type.

        Args:
            event_type: Optional event type to filter by

        Returns:
            List of events
        """
        if event_type is None:
            return self._events.copy()
        return [e for e in self._events if (e.get("type") or e.get("event_type")) == event_type]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of logged events."""
        event_counts: dict[str, int] = {}
        for event in self._events:
            event_type = event.get("type") or event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self._events),
            "event_counts": event_counts,
        }

    def clear(self) -> None:
        """Clear all logged events."""
        self._events.clear()
        self._sequence = 0

    def _normalize_role(self, role: str) -> str:
        if not self.normalize_roles:
            return role
        return _NEUTRAL_ROLE.get(role, role)

    def _normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.normalize_roles or data.get("source") not in _NEUTRAL_SOURCE:
            return data
        normalized = dict(data)
        normalized["source"] = _NEUTRAL_SOURCE[normalized["source"]]
        return normalized

    @staticmethod
    def _to_legacy_event(event: dict[str, Any]) -> dict[str, Any]:
        legacy = dict(event)
        legacy.pop("provider", None)
        if legacy.get("user") in _LEGACY_ROLE:
            legacy["user"] = _LEGACY_ROLE[legacy["user"]]
        data = legacy.get("data")
        if isinstance(data, dict) and data.get("source") in _LEGACY_SOURCE:
            legacy["data"] = dict(data)
            legacy["data"]["source"] = _LEGACY_SOURCE[data["source"]]
        return legacy


class ElevenLabsEventLogger(UserSimulatorEventLogger):
    """Backward-compatible logger preserving the historical event schema."""

    def __init__(self, output_path: Path):
        super().__init__(
            output_path,
            provider="elevenlabs",
            normalize_roles=False,
            include_provider=False,
        )
