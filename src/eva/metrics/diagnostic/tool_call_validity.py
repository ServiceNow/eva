"""Measures the fraction of tool calls that are valid.

Catches incorrect tool calls (wrong tool name, missing/malformed parameters,
invalid enum values, wrong types) but not business-logic errors (reservation
not found, no seats available, etc.).

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from eva.assistant.tools.airline_params import FIELD_ERROR_TYPES
from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore

# Infrastructure errors from ToolExecutor + generic Pydantic fallback.
_TOOL_EXECUTOR_ERROR_TYPES = frozenset(
    {
        "tool_not_found",
        "function_not_found",
        "execution_error",
        "invalid_parameter",
    }
)

# Validation error types derived from Pydantic param models.
_VALIDATION_ERROR_TYPES = frozenset(error_type for error_type, _ in FIELD_ERROR_TYPES.values())

CALL_ERROR_TYPES = _TOOL_EXECUTOR_ERROR_TYPES | _VALIDATION_ERROR_TYPES


@register_metric
class ToolCallValidity(CodeMetric):
    """Fraction of tool calls that are valid (correct tool name, parameters, types).

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "tool_call_validity"
    description = "Debug metric: fraction of tool calls with correctly formatted parameters"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        if not context.tool_responses:
            return MetricScore(
                name=self.name,
                score=1.0,
                normalized_score=1.0,
                details={"total_tool_calls": 0, "note": "No tool calls to evaluate"},
                sub_metrics={
                    "num_tool_calls": MetricScore(
                        name=f"{self.name}.num_tool_calls",
                        score=0.0,
                        normalized_score=None,
                        details={},
                    )
                },
            )

        format_errors = []
        for i, resp in enumerate(context.tool_responses):
            tool_response = resp.get("tool_response", {})
            if not isinstance(tool_response, dict):
                continue

            error_type = tool_response.get("error_type", "")
            if error_type in CALL_ERROR_TYPES:
                params = context.tool_params[i] if i < len(context.tool_params) else {}
                format_errors.append(
                    {
                        "tool_name": resp.get("tool_name"),
                        "error_type": error_type,
                        "message": tool_response.get("message", ""),
                        "parameters": params.get("tool_parameters", {}),
                    }
                )

        total = len(context.tool_responses)
        correct = total - len(format_errors)
        score = correct / total

        sub_metrics = _build_tool_call_validity_sub_metrics(self.name, total, format_errors)

        return MetricScore(
            name=self.name,
            score=round(score, 4),
            normalized_score=round(score, 4),
            details={
                "total_tool_calls": total,
                "valid_tool_calls": correct,
                "invalid_tool_calls": len(format_errors),
                "errors": format_errors,
            },
            sub_metrics=sub_metrics or None,
        )


def _build_tool_call_validity_sub_metrics(
    parent_name: str,
    total_tool_calls: int,
    format_errors: list[dict],
) -> dict[str, MetricScore]:
    """Build sub-metrics for total tool call count and per-error-type rates.

    ``num_tool_calls`` is a count (normalized_score=None). Per-error-type rates
    are emitted for every known error type so keys are stable across records.
    """
    sub_metrics: dict[str, MetricScore] = {
        "num_tool_calls": MetricScore(
            name=f"{parent_name}.num_tool_calls",
            score=float(total_tool_calls),
            normalized_score=None,
            details={},
        )
    }

    if total_tool_calls == 0:
        return sub_metrics

    error_counts: dict[str, int] = dict.fromkeys(CALL_ERROR_TYPES, 0)
    for error in format_errors:
        error_type = error.get("error_type")
        if error_type in error_counts:
            error_counts[error_type] += 1

    for error_type, count in error_counts.items():
        sub_metrics[f"{error_type}_rate"] = make_rate_sub_metric(
            parent_name=parent_name,
            key=f"{error_type}_rate",
            numerator=count,
            denominator=total_tool_calls,
            details={"count": count, "total_tool_calls": total_tool_calls},
            precision=4,
        )

    return sub_metrics
