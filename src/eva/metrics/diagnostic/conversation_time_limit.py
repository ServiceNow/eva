"""Conversation time limit diagnostic metric."""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class ConversationTimeLimitExceededMetric(CodeMetric):
    """1.0 when the conversation finished within the time limit; 0.0 otherwise."""

    name = "conversation_completed_on_time"
    version = "v0.1"
    description = "Diagnostic metric: 1.0 when conversation finished within time limit, 0.0 otherwise"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            reason = context.conversation_ended_reason
            # Note that `time_limit_exceeded` is treated differently from `inactivity_timeout`. `inactivity_timeout`
            # indicates that there was a problem with the simulation whereas `time_limit_exceeded` indicates
            # that the model could not complete the conversation on time.
            timed_out = reason == "time_limit_exceeded"
            score = 0.0 if timed_out else 1.0

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details={
                    "conversation_ended_reason": reason,
                    "timed_out": timed_out,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
