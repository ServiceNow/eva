"""Delegation-rate diagnostic metric.

Measures how often the frontend voice model hands a user turn to the backend (delegation model), for
frameworks with explicit delegation (currently ``openai_live``). The unit is the user turn: assistant turns
include the greeting, backchannels and acknowledgements, which would dilute the rate.

Reads ``framework_logs.jsonl`` directly. A ``delegation_session`` record marks a run as delegation-capable;
without it (any other framework) the metric is skipped. Each ``delegation`` record belongs to the latest user
turn that started at or before it.

Diagnostic metric — reported for tuning the frontend prompt, not used in final pass/fail scores. Neither a
high nor a low rate is good or bad by itself.
"""

import json
from bisect import bisect_right
from pathlib import Path

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


def _skipped(name: str, reason: str) -> MetricScore:
    return MetricScore(name=name, score=None, normalized_score=None, skipped=True, details={"reason": reason})


@register_metric
class DelegationRateMetric(CodeMetric):
    """Fraction of user turns that triggered at least one delegation to the backend model.

    Score: delegated user turns / user turns (0-1).
    """

    name = "delegation_rate"
    category = "diagnostic"
    description = "Diagnostic metric: fraction of user turns delegated to the backend model (delegation frameworks)"
    exclude_from_pass_at_k = True
    version = "v0.1"

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            log_path = Path(context.output_dir) / "framework_logs.jsonl"
            if not log_path.is_file():
                return _skipped(self.name, "No framework_logs.jsonl for this record")

            entries = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))

            if not any(e.get("type") == "delegation_session" for e in entries):
                return _skipped(self.name, "Framework does not use delegation (no delegation_session record)")

            turn_starts = sorted(e["timestamp"] for e in entries if e.get("type") == "turn_start")
            delegations = sorted(e["timestamp"] for e in entries if e.get("type") == "delegation")
            if not turn_starts:
                return _skipped(self.name, "No user turns recorded")

            delegated_turns: set[int] = set()
            unattributed = 0
            for ts in delegations:
                turn_index = bisect_right(turn_starts, ts) - 1
                if turn_index < 0:
                    unattributed += 1
                else:
                    delegated_turns.add(turn_index)

            return MetricScore(
                name=self.name,
                score=len(delegated_turns) / len(turn_starts),
                normalized_score=None,
                details={
                    "num_user_turns": len(turn_starts),
                    "num_delegated_turns": len(delegated_turns),
                    "num_delegations": len(delegations),
                    "num_unattributed_delegations": unattributed,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
