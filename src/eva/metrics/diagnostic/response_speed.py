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


def _load_audit_stats(output_dir: str) -> dict | None:
    """Read audit_log.json once and compute LLM-call / tool-call stats.

    - ``num_llm_calls``: number of LLM completions (``llm_prompts`` entries)
    - ``num_turns``: user turns (``transcript`` entries with ``message_type == "user"``)
    - ``num_tool_calls``: total tool calls across all completions
      (sum of each completion's ``response_message.tool_calls`` length)
    - ``calls_with_tool_calls`` / ``parallel_tool_call_completions``: completions
      emitting >=1 / >1 tool call (the latter reveals parallel/batched tool calls)

    Uses the same LLM-call and turn definitions as the cross-run scan so numbers
    are comparable. Returns a stats dict, or None if the audit log is
    missing/unreadable, has no user turns, or has no ``llm_prompts`` — the last
    case covers models (e.g. GPT realtime / speech-to-speech) that don't log
    prompt-level calls, for which these stats are undefined and must be omitted
    rather than reported as zero.
    """
    audit_path = Path(output_dir) / "audit_log.json"
    if not audit_path.exists():
        return None

    try:
        audit = json.loads(audit_path.read_text())
    except Exception:
        return None

    prompts = audit.get("llm_prompts")
    prompts = prompts if isinstance(prompts, list) else []
    num_calls = len(prompts)
    num_turns = sum(1 for e in audit.get("transcript", []) if isinstance(e, dict) and e.get("message_type") == "user")
    if num_turns <= 0 or num_calls <= 0:
        return None

    per_call_tools = []
    for c in prompts:
        rm = c.get("response_message") if isinstance(c, dict) else None
        tool_calls = rm.get("tool_calls") if isinstance(rm, dict) else None
        per_call_tools.append(len(tool_calls) if isinstance(tool_calls, list) else 0)
    num_tool_calls = sum(per_call_tools)

    return {
        "num_llm_calls": num_calls,
        "num_turns": num_turns,
        "llm_calls_per_turn": round(num_calls / num_turns, 3),
        "num_tool_calls": num_tool_calls,
        "tools_per_llm_call": round(num_tool_calls / num_calls, 3),
        "calls_with_tool_calls": sum(1 for n in per_call_tools if n >= 1),
        "parallel_tool_call_completions": sum(1 for n in per_call_tools if n > 1),
        "max_tools_per_call": max(per_call_tools) if per_call_tools else 0,
    }


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
    version = "v0.3"

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

            # LLM-call / tool-call efficiency diagnostics from audit_log.json.
            audit_stats = _load_audit_stats(context.output_dir)
            if audit_stats is not None:
                # LLM completions per user turn (tool-loop / model efficiency).
                sub_metrics["llm_calls_per_turn"] = MetricScore(
                    name=f"{self.name}.llm_calls_per_turn",
                    score=audit_stats["llm_calls_per_turn"],
                    normalized_score=None,
                    details=audit_stats,
                )
                # Tool calls per LLM completion (reveals sequential vs batched tool calls).
                sub_metrics["tools_per_llm_call"] = MetricScore(
                    name=f"{self.name}.tools_per_llm_call",
                    score=audit_stats["tools_per_llm_call"],
                    normalized_score=None,
                    details=audit_stats,
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
