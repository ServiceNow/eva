"""Response speed metric measuring latency between user and assistant.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

import json
from pathlib import Path

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore
from eva.utils.log_processing import load_audit_log


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


def _load_inference_tool_counts(output_dir: str) -> list[int] | None:
    """Read audit_log.json once and return per-inference tool-call counts.

    An *inference* is one model round-trip (one ``llm_prompts`` entry): the
    agent may run several within a single user turn as it loops through the
    reason-act cycle. Returns a list where element ``i`` is the number of tool
    calls emitted by the i-th inference; the caller derives all aggregate counts
    and rates from it.

    Uses the same inference definition as the cross-run scan so numbers are
    comparable. Returns None if the audit log is missing/unreadable or has no
    ``llm_prompts``.
    """
    audit = load_audit_log(Path(output_dir) / "audit_log.json")
    if audit is None:
        return None

    prompts = audit.get("llm_prompts")
    prompts = prompts if isinstance(prompts, list) else []
    if not prompts:
        return None

    per_inference_tools = []
    for c in prompts:
        rm = c.get("response_message") if isinstance(c, dict) else None
        tool_calls = rm.get("tool_calls") if isinstance(rm, dict) else None
        per_inference_tools.append(len(tool_calls) if isinstance(tool_calls, list) else 0)

    return per_inference_tools


def _tool_call_turn_ids(context: MetricContext) -> set:
    """Return the set of turn_ids that contain at least one tool call."""
    return {entry["turn_id"] for entry in (context.conversation_trace or []) if entry.get("type") == "tool_call"}


def _split_by_tool_calls(
    context: MetricContext,
) -> tuple[list[float], list[float]]:
    """Partition per_turn_latency values into (with_tool_calls, no_tool_calls)."""
    tool_call_turn_ids = _tool_call_turn_ids(context)

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
    version = "v0.4"

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

            # Inference / tool-call efficiency diagnostics from audit_log.json.
            # Rates count tool-calling inferences only (no-tool inferences excluded).
            # Each sub-metric carries only the counts that explain its own score.
            per_inference_tools = _load_inference_tool_counts(context.output_dir)
            if per_inference_tools is not None:
                num_inferences_with_tool_calls = sum(1 for n in per_inference_tools if n >= 1)

                # Tool calls per tool-using turn: when the model uses tools in a turn,
                # how many tool calls land in that turn (across all its inferences).
                num_turns_with_tool_calls = len(_tool_call_turn_ids(context))
                sub_metrics["num_tool_calls_per_tool_turn"] = MetricScore(
                    name=f"{self.name}.num_tool_calls_per_tool_turn",
                    score=(
                        round(sum(per_inference_tools) / num_turns_with_tool_calls, 3)
                        if num_turns_with_tool_calls
                        else 0.0
                    ),
                    normalized_score=None,
                    details={
                        "num_tool_calls": sum(per_inference_tools),
                        "num_turns_with_tool_calls": num_turns_with_tool_calls,
                    },
                )
                # Mean tool calls per tool-calling inference — the model's tool-call
                # parallelism (1.0 = always sequential, >1.0 = emits parallel tool calls).
                sub_metrics["tool_call_parallelism"] = MetricScore(
                    name=f"{self.name}.tool_call_parallelism",
                    score=(
                        round(sum(per_inference_tools) / num_inferences_with_tool_calls, 3)
                        if num_inferences_with_tool_calls
                        else 0.0
                    ),
                    normalized_score=None,
                    details={
                        "num_tool_calls": sum(per_inference_tools),
                        "num_inferences_with_tool_calls": num_inferences_with_tool_calls,
                        "num_inferences_with_parallel_tool_calls": sum(1 for n in per_inference_tools if n > 1),
                        "max_tools_per_inference": max(per_inference_tools),
                    },
                )
                # Fraction of inferences that call >=1 tool.
                sub_metrics["tool_inference_rate"] = MetricScore(
                    name=f"{self.name}.tool_inference_rate",
                    score=round(num_inferences_with_tool_calls / len(per_inference_tools), 3),
                    normalized_score=None,
                    details={
                        "num_inferences_with_tool_calls": num_inferences_with_tool_calls,
                        "num_inferences": len(per_inference_tools),
                    },
                )

            # Add per-component latency sub_metrics from result.json.
            # result.json stores these in milliseconds; convert to seconds here
            # to match the top-level response_speed score (already in seconds).
            for key, latency_stats in _load_component_latencies(context.output_dir).items():
                seconds_details = {
                    (f"{stat_key[:-3]}_seconds" if stat_key.endswith("_ms") else stat_key): (
                        round(stat_value / 1000, 3)
                        if stat_key.endswith("_ms") and stat_value is not None
                        else stat_value
                    )
                    for stat_key, stat_value in latency_stats.items()
                }
                sub_metrics[key] = MetricScore(
                    name=f"{self.name}.{key}",
                    score=round(latency_stats["mean_ms"] / 1000, 3),
                    normalized_score=None,
                    details=seconds_details,
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
