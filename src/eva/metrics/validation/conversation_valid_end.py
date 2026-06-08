"""Conversation-valid-end validation metric."""

import json
from pathlib import Path

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore
from eva.user_simulator.events import resolve_user_simulator_events_path


@register_metric
class ConversationValidEndMetric(CodeMetric):
    """Binary score: 1.0 when the conversation ended on goodbye OR agent-timeout-on-user-turn; 0.0 otherwise."""

    name = "conversation_valid_end"
    version = "v0.2"
    description = "Validation metric: conversation reached a definitive end state"
    category = "validation"

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            agent_timeout = is_agent_timeout_on_user_turn(
                context.conversation_ended_reason,
                context.audio_timestamps_user_turns,
                context.audio_timestamps_assistant_turns,
            )
            if agent_timeout:
                return MetricScore(
                    name=self.name,
                    score=1.0,
                    normalized_score=1.0,
                    details={
                        "ended_properly": True,
                        "reason": "agent_timeout_on_user_turn",
                        "details": "agent timed out on the user's final turn (definitive terminal state)",
                    },
                )
            output_dir = Path(context.output_dir)
            events_path = resolve_user_simulator_events_path(output_dir)

            if events_path is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="User simulator events file not found",
                    details={"output_dir": str(output_dir)},
                )

            with open(events_path) as f:
                lines = f.readlines()

            if not lines:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"{events_path.name} is empty",
                    details={"file_path": str(events_path)},
                )

            last_line = lines[-1].strip()
            if not last_line:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"Last line in {events_path.name} is empty",
                    details={"file_path": str(events_path)},
                )

            try:
                last_event = json.loads(last_line)
            except json.JSONDecodeError as e:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"Failed to parse last line as JSON: {e}",
                    details={"file_path": str(events_path), "last_line": last_line},
                )

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
                        "file_path": str(events_path),
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
                        "file_path": str(events_path),
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
                    "file_path": str(events_path),
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
