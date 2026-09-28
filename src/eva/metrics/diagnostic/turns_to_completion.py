"""Turns-to-completion diagnostic metric.

Measures how many conversation turns a task took, reported only for
conversations where the task was actually completed. This makes the value a
meaningful "how many turns did it take to succeed" signal rather than mixing in
the turn counts of failed conversations.

Task completion is determined the same way as the ``task_completion`` accuracy
metric: the session must be authenticated correctly and the final scenario
database state must match the expected state (SHA-256 hash comparison).

Diagnostic metric — reported for benchmarking, not used in final pass/fail
scores. Lower is better.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.diagnostic.task_completion_utils import is_task_completed
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class TurnsToCompletionMetric(CodeMetric):
    """Number of conversation turns to complete the task.

    Reports ``context.num_turns`` (total conversation turns) when the task was
    completed, and is skipped otherwise so that efficiency stats only aggregate
    over successful conversations.

    Score: total number of conversation turns (lower is better).
    """

    name = "turns_to_completion"
    category = "diagnostic"
    description = "Diagnostic metric: number of conversation turns to complete the task (successful runs only)"
    exclude_from_pass_at_k = True
    higher_is_better = False  # Score is a turn count — lower is better.

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            completed, reason = is_task_completed(context)

            if not completed:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={"task_completed": False, "reason": reason},
                )

            num_turns = context.num_turns
            if not num_turns or num_turns <= 0:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={
                        "task_completed": True,
                        "reason": f"No valid turn count recorded (num_turns={num_turns})",
                    },
                )

            return MetricScore(
                name=self.name,
                score=float(num_turns),
                normalized_score=None,
                details={
                    "task_completed": True,
                    "num_turns": num_turns,
                    "reason": "Task completed — reporting total conversation turns",
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
