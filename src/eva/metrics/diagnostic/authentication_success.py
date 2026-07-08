"""Authentication success metric - checks if the session was authenticated correctly.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from collections import Counter

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


def _normalize_session_value(v: object) -> object:
    """Normalize a session value for comparison — strings are lowercased."""
    return v.lower() if isinstance(v, str) else v


def compute_session_auth_mismatches(expected_scenario_db: dict, final_scenario_db: dict) -> dict:
    """Check whether the final DB session satisfies the expected session.

    String values are compared case-insensitively.
    Returns a dict of mismatched keys (empty dict means auth succeeded or no auth expected).
    """
    expected_session = expected_scenario_db.get("session", {})
    actual_session = final_scenario_db.get("session", {})
    return {
        k: {"expected": v, "actual": actual_session.get(k)}
        for k, v in expected_session.items()
        if _normalize_session_value(actual_session.get(k)) != _normalize_session_value(v)
    }


@register_metric
class AuthenticationSuccessMetric(CodeMetric):
    """Checks whether the agent successfully authenticated the user.

    Compares the 'session' key in the final scenario database against the
    expected session in the ground truth. Authentication is successful if the
    final session is a superset of the expected session — i.e., every key-value
    pair in expected_session is present in the actual final session.

    Score: 1.0 if final session is a superset of expected session, 0.0 otherwise.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "authentication_success"
    version = "v0.1"
    description = "Checks if session state in final DB is a superset of expected session"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute authentication success from final scenario database session state."""
        try:
            expected_session = context.expected_scenario_db.get("session", {})
            actual_session = context.final_scenario_db.get("session", {})

            if not expected_session:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details={"reason": "No expected session to verify — skipping auth check"},
                )

            mismatches = compute_session_auth_mismatches(context.expected_scenario_db, context.final_scenario_db)
            success = len(mismatches) == 0

            sub_metrics = _build_authentication_success_sub_metrics(
                self.name, context.tool_responses, context.agent_tools, success
            )

            return MetricScore(
                name=self.name,
                score=1.0 if success else 0.0,
                normalized_score=1.0 if success else 0.0,
                details={
                    "expected_session": expected_session,
                    "actual_session": actual_session,
                    "mismatches": mismatches,
                    "reason": "Authentication successful"
                    if success
                    else f"Session mismatch on keys: {list(mismatches)}",
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)


def _build_authentication_success_sub_metrics(
    parent_name: str,
    tool_responses: list[dict],
    agent_tools: list[dict],
    auth_success: bool,
) -> dict[str, MetricScore]:
    """Build sub-metrics for authentication tool call behaviour."""
    auth_tool_names = {t.get("name") for t in agent_tools if t.get("tool_type") == "auth" and t.get("name")}
    auth_calls = [r for r in tool_responses if r.get("tool_name") in auth_tool_names]
    if not auth_calls:
        return {}

    sub_metrics: dict[str, MetricScore] = {}

    counts_per_tool = Counter(r.get("tool_name") for r in auth_calls)
    avg_calls = sum(counts_per_tool.values()) / len(counts_per_tool) if counts_per_tool else 0.0

    if auth_success:
        first_try_success = all(count == 1 for count in counts_per_tool.values())
        sub_metrics["auth_first_try_success"] = MetricScore(
            name=f"{parent_name}.auth_first_try_success",
            score=1.0 if first_try_success else 0.0,
            normalized_score=1.0 if first_try_success else 0.0,
            details={"calls_per_tool": counts_per_tool, "num_auth_tools": len(counts_per_tool)},
        )
        sub_metrics["auth_num_calls"] = MetricScore(
            name=f"{parent_name}.auth_num_calls",
            score=round(avg_calls, 4),
            normalized_score=None,
            details={"calls_per_tool": counts_per_tool, "num_auth_tools": len(counts_per_tool)},
        )
    else:
        sub_metrics["auth_num_calls_on_failure"] = MetricScore(
            name=f"{parent_name}.auth_num_calls_on_failure",
            score=round(avg_calls, 4),
            normalized_score=None,
            details={"calls_per_tool": counts_per_tool, "num_auth_tools": len(counts_per_tool)},
        )

    return sub_metrics
