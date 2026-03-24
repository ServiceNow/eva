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
        )
