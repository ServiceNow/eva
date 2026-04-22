"""Conversation-correctly-finished diagnostic metric (previously ``agent_turn_response``).

Flags records where the conversation ended with ``inactivity_timeout`` and the user was
the last speaker by audio timeline — i.e. the agent failed to respond to the user's
final turn. Not directly used in final evaluation scores; excluded from pass@k.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.processor import is_agent_timeout_on_user_turn, last_audio_speaker
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class ConversationCorrectlyFinishedMetric(CodeMetric):
    """Scores whether the agent responded on every user turn.

    Score: 1.0 if the agent responded to every user turn (conversation ended normally
    or the assistant spoke last); 0.0 if the conversation ended with
    ``inactivity_timeout`` and the user was the last speaker (agent failed to respond
    to the user's final turn).
    """

    name = "conversation_correctly_finished"
    description = "Diagnostic metric: 0.0 when agent failed to respond to the user's final turn"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            reason = context.conversation_ended_reason
            speaker = last_audio_speaker(
                context.audio_timestamps_user_turns,
                context.audio_timestamps_assistant_turns,
            )

            missed_turn = is_agent_timeout_on_user_turn(
                reason,
                context.audio_timestamps_user_turns,
                context.audio_timestamps_assistant_turns,
            )
            score = 0.0 if missed_turn else 1.0

            if missed_turn:
                human_reason = "conversation ended with inactivity_timeout and user was the last speaker"
            elif reason == "inactivity_timeout":
                human_reason = f"inactivity_timeout but last speaker was {speaker!r}"
            else:
                human_reason = f"conversation ended with reason={reason!r}"

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details={
                    "conversation_ended_reason": reason,
                    "last_audio_speaker": speaker,
                    "reason": human_reason,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
