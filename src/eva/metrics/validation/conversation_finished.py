"""Conversation finished validation metric."""

import json
from pathlib import Path

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class ConversationFinishedMetric(CodeMetric):
    """Conversation finished validation metric.

    Checks that the conversation properly ended with an end_call tool.
    Reads the elevenlabs_events.jsonl file and verifies the last entry
    is a tool_response with tool_name == "end_call".

    Binary score: 1.0 (properly ended), 0.0 (did not end properly)
    """

    name = "conversation_finished"
    description = "Validation metric for checking if conversation ended with end_call tool"
    category = "validation"

    async def compute(self, context: MetricContext) -> MetricScore:
        """Check if conversation properly ended with end_call."""
        try:
            output_dir = Path(context.output_dir)
            elevenlabs_events_path = output_dir / "elevenlabs_events.jsonl"

            if not elevenlabs_events_path.exists():
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="elevenlabs_events.jsonl file not found",
                    details={"file_path": str(elevenlabs_events_path)},
                )

            with open(elevenlabs_events_path) as f:
                lines = f.readlines()

            if not lines:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="elevenlabs_events.jsonl is empty",
                    details={"file_path": str(elevenlabs_events_path)},
                )

            last_line = lines[-1].strip()
            if not last_line:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Last line in elevenlabs_events.jsonl is empty",
                    details={"file_path": str(elevenlabs_events_path)},
                )

            try:
                last_event = json.loads(last_line)
            except json.JSONDecodeError as e:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"Failed to parse last line as JSON: {e}",
                    details={"file_path": str(elevenlabs_events_path), "last_line": last_line},
                )

            # Check if type is "tool_response"
            event_type = last_event.get("type")
            if event_type != "connection_state":
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    details={
                        "ended_properly": False,
                        "last_event_type": event_type,
                        "reason": f"Last event type is '{event_type}', expected 'connection_state'",
                        "file_path": str(elevenlabs_events_path),
                    },
                )

            data = last_event.get("data", {})
            details = data.get("details", {})
            reason = details.get("reason")

            if reason != "goodbye":
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    details={
                        "ended_properly": False,
                        "last_event_type": event_type,
                        "reason": "conversation ended for unknown reasons",
                        "file_path": str(elevenlabs_events_path),
                    },
                )

            return MetricScore(
                name=self.name,
                score=1.0,
                normalized_score=1.0,
                details={
                    "ended_properly": True,
                    "last_event_type": event_type,
                    "details": "end_call was called successfully",
                    "file_path": str(elevenlabs_events_path),
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
