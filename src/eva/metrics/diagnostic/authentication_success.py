"""Authentication success metric - checks if the session was authenticated correctly.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


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
                    score=1.0,
                    normalized_score=1.0,
                    details={"reason": "No expected session to verify — skipping auth check"},
                )

            # Check superset: every key-value in expected_session must be in actual_session
            mismatches = {
                k: {"expected": v, "actual": actual_session.get(k)}
                for k, v in expected_session.items()
                if actual_session.get(k) != v
            }

            success = len(mismatches) == 0

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
            )

        except Exception as e:
            return self._handle_error(e, context)
