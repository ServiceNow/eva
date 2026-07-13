"""Response speed metric measuring latency between user and assistant.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

import json
from pathlib import Path

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


def _load_component_latencies(output_dir: str) -> dict[str, dict]:
    """Load per-component latency stats from result.json.

    Returns a dict mapping short keys (e.g. "llm_latency", "stt_latency",
    "tts_latency") to their stats dicts, only for non-null entries.
    """
    result_path = Path(output_dir) / "result.json"
    if not result_path.exists():
        return {}

    try:
        result_data = json.loads(result_path.read_text())
    except Exception:
        return {}

    latencies: dict[str, dict] = {}
    for key in ("llm_latency", "stt_latency", "tts_latency"):
        value = result_data.get(key)
        if value is not None and isinstance(value, dict) and value.get("mean_ms") is not None:
            latencies[key] = value

    return latencies


def _split_by_tool_calls(
    context: MetricContext,
) -> tuple[list[float], list[float]]:
    """Partition per_turn_latency values into (with_tool_calls, no_tool_calls)."""
    tool_call_turn_ids = {
        entry["turn_id"] for entry in (context.conversation_trace or []) if entry.get("type") == "tool_call"
    }

    with_tool = [v for k, v in context.latency_assistant_turns.items() if k in tool_call_turn_ids]
    no_tool = [v for k, v in context.latency_assistant_turns.items() if k not in tool_call_turn_ids]

    return with_tool, no_tool


def _compute_speed_stats(latencies: list[float]) -> dict | None:
    """Compute summary stats for a list of latencies, applying the sanity filter.

    Returns None if no valid values remain after filtering.
    """
    valid = [v for v in latencies if 0 < v < 1000]
    if not valid:
        return None
    return {
        "mean_speed_seconds": round(sum(valid) / len(valid), 3),
        "max_speed_seconds": round(max(valid), 3),
        "num_turns": len(valid),
        "per_turn_speeds": [round(v, 3) for v in valid],
    }


@register_metric
class ResponseSpeedMetric(CodeMetric):
    """Response speed metric.

    Measures the elapsed time between the end of the user's utterance
    and the beginning of the assistant's response, using per_turn_latency
    from the turn_taking metric.

    Reports raw latency values in seconds — no normalization applied.

    Details include a breakdown by turns with and without tool calls.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "response_speed"
    category = "diagnostic"
    description = "Diagnostic metric: latency between user utterance end and assistant response start"
    exclude_from_pass_at_k = True
    higher_is_better = False  # Score is latency in seconds — lower is better.
    version = "v0.2"

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            if not context.latency_assistant_turns:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                )

            all_latencies = list(context.latency_assistant_turns.values())
            overall_stats = _compute_speed_stats(all_latencies)

            if not overall_stats:
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                )

            dropped = [v for v in all_latencies if not (0 < v < 1000)]
            if dropped:
                self.logger.warning(
                    f"[{context.record_id}] Dropped {len(dropped)} unusual response speed(s): {dropped}"
                )

            with_tool, no_tool = _split_by_tool_calls(context)

            sub_metrics: dict[str, MetricScore] = {}
            for key, latencies in (("with_tool_calls", with_tool), ("no_tool_calls", no_tool)):
                stats = _compute_speed_stats(latencies)
                if stats is not None:
                    sub_metrics[key] = MetricScore(
                        name=f"{self.name}.{key}",
                        score=stats["mean_speed_seconds"],
                        normalized_score=None,
                        details=stats,
                    )

            # Add per-component latency sub_metrics from result.json
            for key, latency_stats in _load_component_latencies(context.output_dir).items():
                sub_metrics[key] = MetricScore(
                    name=f"{self.name}.{key}",
                    score=latency_stats["mean_ms"],
                    normalized_score=None,
                    details=latency_stats,
                )

            return MetricScore(
                name=self.name,
                score=overall_stats["mean_speed_seconds"],
                normalized_score=None,
                details=overall_stats,
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
