"""Authentication success metric - checks if get_reservation was called successfully.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class AuthenticationSuccessMetric(CodeMetric):
    """Checks whether the agent successfully authenticated the user via get_reservation.

    Looks at tool_responses for entries where tool_name == "get_reservation"
    and checks if any had tool_response.status == "success".

    Score: 1.0 if at least one get_reservation call succeeded, 0.0 otherwise.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "authentication_success"
    description = "Debug metric: checks if get_reservation was called with a successful result"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute authentication success from tool responses."""
        try:
            tool_responses = context.tool_responses or []

            get_reservation_calls = [resp for resp in tool_responses if resp.get("tool_name") == "get_reservation"]

            found = len(get_reservation_calls) > 0

            success_count = 0
            for resp in get_reservation_calls:
                tool_response = resp.get("tool_response", {})
                if isinstance(tool_response, dict) and tool_response.get("status") == "success":
                    success_count += 1

            score = 1.0 if success_count > 0 else 0.0

            # Determine reason for failure
            if not found:
                reason = "get_reservation tool was never called"
            elif success_count == 0:
                reason = "get_reservation was called but never returned status success"
            else:
                reason = "get_reservation called successfully"

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details={
                    "get_reservation_found": found,
                    "get_reservation_call_count": len(get_reservation_calls),
                    "get_reservation_success_count": success_count,
                    "reason": reason,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
