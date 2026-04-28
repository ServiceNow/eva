"""EVA composite metric definitions and aggregation logic.

Single source of truth for all EVA composite scores (EVA-A, EVA-X, EVA-overall).
Edit EVA_COMPOSITES to change which metrics are included or how they are combined.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from eva.models.results import RecordMetrics
from eva.utils.pass_at_k import (
    compute_pass_at_k,
    compute_pass_power_k,
    parse_trial_record_id,
)


@dataclass
class EVACompositeDefinition:
    """Definition of a single EVA composite metric."""

    name: str
    component_metrics: list[str]
    aggregation_type: Literal["pass", "mean", "derived"]
    # For "pass" type: metric -> (operator, threshold)
    thresholds: dict[str, tuple[str, float]] = field(default_factory=dict)
    # For "derived" type: list of prerequisite composite names that must all be 1.0
    derived_from: list[str] = field(default_factory=list)


# ── Composite definitions ────────────────────────────────────────────
EVA_COMPOSITES: list[EVACompositeDefinition] = [
    EVACompositeDefinition(
        name="EVA-A_pass",
        component_metrics=["task_completion", "faithfulness", "agent_speech_fidelity"],
        aggregation_type="pass",
        thresholds={
            "task_completion": ("==", 1.0),
            "faithfulness": (">=", 0.5),
            "agent_speech_fidelity": (">=", 0.95),
        },
    ),
    EVACompositeDefinition(
        name="EVA-X_pass",
        component_metrics=["conversation_progression", "turn_taking", "conciseness"],
        aggregation_type="pass",
        thresholds={
            "conversation_progression": (">=", 0.5),
            "turn_taking": (">=", 0.5),
            "conciseness": (">=", 0.5),
        },
    ),
    EVACompositeDefinition(
        name="EVA-A_mean",
        component_metrics=["task_completion", "faithfulness", "agent_speech_fidelity"],
        aggregation_type="mean",
    ),
    EVACompositeDefinition(
        name="EVA-X_mean",
        component_metrics=["conversation_progression", "conciseness", "turn_taking"],
        aggregation_type="mean",
    ),
    EVACompositeDefinition(
        name="EVA-overall_mean",
        component_metrics=[
            "task_completion",
            "faithfulness",
            "agent_speech_fidelity",
            "conversation_progression",
            "conciseness",
            "turn_taking",
        ],
        aggregation_type="mean",
    ),
    EVACompositeDefinition(
        name="EVA-overall_pass",
        component_metrics=[],
        aggregation_type="derived",
        derived_from=["EVA-A_pass", "EVA-X_pass"],
    ),
]


def _check_threshold(value: float, operator: str, threshold: float) -> bool:
    """Check whether a value passes the given threshold comparison."""
    if operator == "==":
        return math.isclose(value, threshold, abs_tol=1e-9)
    if operator == ">=":
        return value >= threshold or math.isclose(value, threshold, abs_tol=1e-9)
    if operator == ">":
        return value > threshold and not math.isclose(value, threshold, abs_tol=1e-9)
    raise ValueError(f"Unknown operator: {operator}")


def compute_record_aggregates(
    record_metrics: RecordMetrics,
    composites: list[EVACompositeDefinition] | None = None,
) -> dict[str, float | None]:
    """Compute all EVA composite scores for a single record.

    Args:
        record_metrics: The record's individual metric scores.
        composites: Custom composite definitions. Defaults to EVA_COMPOSITES.

    Returns:
        Dict mapping composite name to score (1.0/0.0 for pass, float for mean,
        None if required components are missing).
    """
    composites = composites or EVA_COMPOSITES
    results: dict[str, float | None] = {}

    for comp in composites:
        if comp.aggregation_type == "pass":
            # Components missing or errored collapse the composite to None (genuine data
            # absence). Components flagged as skipped are excluded from the pass check
            # so they don't mask applicable components.
            scores: list[tuple[float, str, float]] = []
            has_error_or_missing = False
            for metric_name in comp.component_metrics:
                metric = record_metrics.metrics.get(metric_name)
                if metric is None or metric.error:
                    has_error_or_missing = True
                    break
                if metric.skipped:
                    continue
                val = metric.normalized_score if metric.normalized_score is not None else metric.score
                op, thresh = comp.thresholds[metric_name]
                scores.append((val, op, thresh))

            if has_error_or_missing or not scores:
                results[comp.name] = None
            else:
                all_pass = all(_check_threshold(v, op, th) for v, op, th in scores)
                results[comp.name] = 1.0 if all_pass else 0.0

        elif comp.aggregation_type == "mean":
            # Average of available components; None if no components available
            available: list[float] = []
            for metric_name in comp.component_metrics:
                val = record_metrics.get_score(metric_name)
                if val is not None:
                    available.append(val)
            results[comp.name] = sum(available) / len(available) if available else None

        elif comp.aggregation_type == "derived":
            # All prerequisite composites must be 1.0
            prereqs = [results.get(name) for name in comp.derived_from]
            if any(p is None for p in prereqs):
                results[comp.name] = None
            else:
                results[comp.name] = 1.0 if all(math.isclose(p, 1.0, abs_tol=1e-9) for p in prereqs) else 0.0

    return results


def compute_run_level_aggregates(
    all_metrics: dict[str, RecordMetrics],
    num_draws: int = 1,
    composites: list[EVACompositeDefinition] | None = None,
) -> dict:
    """Compute run-level aggregate scores from all records.

    Args:
        all_metrics: Dict mapping record ID to RecordMetrics (must have aggregate_metrics populated).
        num_draws: Number of draws (k) for pass@k computation.
        composites: Custom composite definitions. Defaults to EVA_COMPOSITES.

    Returns:
        Dict with per-composite statistics and optional pass@k data.
    """
    composites = composites or EVA_COMPOSITES

    # Collect per-record values for each composite, tracking Nones
    total_records = len(all_metrics)
    composite_values: dict[str, list[float]] = defaultdict(list)
    composite_none_counts: dict[str, int] = defaultdict(int)
    for record_metrics in all_metrics.values():
        agg = record_metrics.aggregate_metrics
        for name, value in agg.items():
            if value is not None:
                composite_values[name].append(value)
            else:
                composite_none_counts[name] += 1

    result: dict = {}

    for comp in composites:
        values = composite_values.get(comp.name, [])
        none_count = composite_none_counts.get(comp.name, 0)
        if not values and none_count == 0:
            continue

        mean_val = sum(values) / len(values) if values else None
        entry: dict = {
            "mean": round(mean_val, 4) if mean_val is not None else None,
            "count": len(values),
            "none_count": none_count,
            "total_records": total_records,
        }

        if values:
            if comp.aggregation_type in ("pass", "derived"):
                entry["success_rate"] = round(mean_val, 4)
            else:
                entry["success_rate"] = round(sum(1 for v in values if v >= 0.5) / len(values), 4)

        result[comp.name] = entry

    # pass_k for aggregate metrics if multi-trial
    if num_draws > 1:
        pass_k_data = _compute_aggregate_pass_k(all_metrics, num_draws, composites)
        if pass_k_data:
            result["pass_k"] = pass_k_data

    return result


def _compute_aggregate_pass_k(
    all_metrics: dict[str, RecordMetrics],
    num_draws: int,
    composites: list[EVACompositeDefinition] | None = None,
) -> dict:
    """Compute pass@1, pass@k, pass^k (observed), and pass^k (theoretical) for aggregate metrics across trials."""
    composites = composites or EVA_COMPOSITES

    # Group records by base ID
    grouped: dict[str, list[tuple[int, RecordMetrics]]] = {}
    for record_id, metrics in all_metrics.items():
        base_id, trial_idx = parse_trial_record_id(record_id)
        if trial_idx is not None:
            grouped.setdefault(base_id, []).append((trial_idx, metrics))

    if not grouped:
        return {}

    # For each composite that is a pass metric, compute pass stats across base records
    pass_composites = [c for c in composites if c.aggregation_type in ("pass", "derived")]
    result: dict = {}

    for comp in pass_composites:
        pass_at_1_values: list[float] = []
        pass_at_k_values: list[float] = []
        pass_power_k_observed_values: list[float] = []
        pass_power_k_theoretical_values: list[float] = []

        for _base_id, trials in grouped.items():
            trials.sort(key=lambda x: x[0])
            trial_values = [rm.aggregate_metrics.get(comp.name) for _, rm in trials]

            # Skip record if any trial is missing/errored for this composite
            if len(trial_values) < num_draws or any(v is None for v in trial_values):
                continue

            n = len(trial_values)
            c = sum(1 for v in trial_values if math.isclose(v, 1.0, abs_tol=1e-9))
            k = min(num_draws, n)

            pass_at_1_values.append(compute_pass_at_k(n, c, 1))
            pass_at_k_values.append(compute_pass_at_k(n, c, k))
            pass_power_k_observed_values.append(compute_pass_power_k(n, c, k))
            pass_power_k_theoretical_values.append((c / n) ** k)

        if pass_at_k_values:
            count = len(pass_at_k_values)
            result[comp.name] = {
                "pass_at_1": round(sum(pass_at_1_values) / count, 4),
                "pass_at_k": round(sum(pass_at_k_values) / count, 4),
                "pass_power_k_observed": round(sum(pass_power_k_observed_values) / count, 4),
                "pass_power_k_theoretical": round(sum(pass_power_k_theoretical_values) / count, 4),
                "k": num_draws,
                "count": count,
            }

    return result
