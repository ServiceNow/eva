"""Time-to-completion diagnostic metric.

Measures the total wall-clock time (in seconds) a conversation took, reported
only for conversations where the task was actually completed. This makes the
value a meaningful "how long did it take to succeed" signal rather than mixing
in the durations of failed conversations.

Note: the duration is wall-clock from conversation start to end (started_at to
ended_at in ConversationResult), which includes connection setup/teardown time
in addition to active audio — not just the spoken audio duration.

Task completion is determined the same way as the ``task_completion`` accuracy
metric: the session must be authenticated correctly and the final scenario
database state must match the expected state (SHA-256 hash comparison).

Also reports, as submetrics, whether the conversation finished within the
configured time limit (``conversation_completed_on_time``) and whether it
specifically ended via a timeout (``timed_out``) — absorbing what used to be
the standalone ``conversation_completed_on_time`` metric.

Diagnostic metric — reported for benchmarking, not used in final pass/fail
scores. Lower is better.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.diagnostic.task_completion_utils import is_task_completed
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class TimeToCompletionMetric(CodeMetric):
    """Wall-clock time (seconds) to complete the task.

    Reports ``context.duration_seconds`` when the task was completed, and is
    skipped otherwise so that efficiency stats only aggregate over successful
    conversations.

    Score: total conversation duration in seconds (lower is better).
    """

    name = "time_to_completion"
    category = "diagnostic"
    description = "Diagnostic metric: wall-clock time in seconds to complete the task (successful runs only)"
    exclude_from_pass_at_k = True
    higher_is_better = False  # Score is time in seconds — lower is better.

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            reason = context.conversation_ended_reason
            # Note that `time_limit_exceeded` is treated differently from `inactivity_timeout`. `inactivity_timeout`
            # indicates that there was a problem with the simulation whereas `time_limit_exceeded` indicates
            # that the model could not complete the conversation on time.
            timed_out = reason == "time_limit_exceeded"
            completed_on_time = 0.0 if timed_out else 1.0

            completed, task_reason = is_task_completed(context)

            if not completed:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={
                        "task_completed": False,
                        "reason": task_reason,
                        "conversation_ended_reason": reason,
                        "conversation_completed_on_time": completed_on_time,
                        "timed_out": timed_out,
                    },
                )

            duration = context.duration_seconds
            if duration is None or duration <= 0:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={
                        "task_completed": True,
                        "reason": f"No valid duration recorded (duration_seconds={duration})",
                        "conversation_ended_reason": reason,
                        "conversation_completed_on_time": completed_on_time,
                        "timed_out": timed_out,
                    },
                )

            return MetricScore(
                name=self.name,
                score=round(duration, 3),
                normalized_score=None,
                details={
                    "task_completed": True,
                    "duration_seconds": round(duration, 3),
                    "reason": "Task completed — reporting total conversation duration",
                    "conversation_ended_reason": reason,
                    "conversation_completed_on_time": completed_on_time,
                    "timed_out": timed_out,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
