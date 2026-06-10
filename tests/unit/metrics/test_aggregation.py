"""Unit tests for EVA composite metric aggregation."""

import numpy as np
import pytest

from eva.metrics.aggregation import (
    EVA_COMPOSITES,
    _check_threshold,
    _scenario_values_for_composite,
    compute_record_aggregates,
    compute_run_level_aggregates,
    scenario_means_for_metric,
)
from eva.metrics.runner import MetricsRunner
from eva.models.results import MetricScore, PassAtKResult, RecordMetrics
from eva.utils.bootstrap import run_seed

from .conftest import make_record_metrics


def _composite_by_name(name: str):
    return next(c for c in EVA_COMPOSITES if c.name == name)


def _make_clean_records(n: int, passing: int) -> dict[str, RecordMetrics]:
    """Return n records, ``passing`` of which pass EVA-A_pass."""
    records: dict[str, RecordMetrics] = {}
    for i in range(n):
        is_pass = i < passing
        r = make_record_metrics(
            {
                "task_completion": 1.0 if is_pass else 0.0,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.8,
                "conciseness": 0.5,
            },
            record_id=f"1.1.{i}",
        )
        r.aggregate_metrics = compute_record_aggregates(r)
        records[f"1.1.{i}"] = r
    return records


class TestCheckThreshold:
    def test_eq_exact(self):
        assert _check_threshold(1.0, "==", 1.0) is True

    def test_eq_close(self):
        assert _check_threshold(1.0 + 1e-10, "==", 1.0) is True

    def test_eq_fail(self):
        assert _check_threshold(0.99, "==", 1.0) is False

    def test_gte_exact(self):
        assert _check_threshold(0.5, ">=", 0.5) is True

    def test_gte_above(self):
        assert _check_threshold(0.6, ">=", 0.5) is True

    def test_gte_below(self):
        assert _check_threshold(0.4, ">=", 0.5) is False

    def test_gt_above(self):
        assert _check_threshold(0.91, ">", 0.9) is True

    def test_gt_exact(self):
        assert _check_threshold(0.9, ">", 0.9) is False

    def test_gt_below(self):
        assert _check_threshold(0.89, ">", 0.9) is False

    def test_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            _check_threshold(1.0, "!=", 1.0)


class TestComputeRecordAggregates:
    def test_all_pass(self):
        """All metrics meet their thresholds."""
        rm = make_record_metrics(
            {
                "task_completion": 1.0,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.8,
                "conciseness": 0.5,
            }
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_pass"] == 1.0
        assert agg["EVA-X_pass"] == 1.0
        assert agg["EVA-overall_pass"] == 1.0
        assert agg["EVA-A_mean"] == pytest.approx((1.0 + 0.5 + 0.95) / 3)
        assert agg["EVA-X_mean"] == pytest.approx((0.5 + 0.8 + 0.5) / 3)
        assert agg["EVA-overall_mean"] == pytest.approx((1.0 + 0.5 + 0.95 + 0.5 + 0.8 + 0.5) / 6)

    def test_eva_a_fails(self):
        """task_completion < 1.0 causes EVA-A_pass to fail."""
        rm = make_record_metrics(
            {
                "task_completion": 0.5,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.8,
                "conciseness": 0.5,
            }
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_pass"] == 0.0
        assert agg["EVA-X_pass"] == 1.0
        assert agg["EVA-overall_pass"] == 0.0

    def test_eva_x_fails(self):
        """Conciseness < 0.5 causes EVA-X_pass to fail."""
        rm = make_record_metrics(
            {
                "task_completion": 1.0,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.8,
                "conciseness": 0.3,
            }
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_pass"] == 1.0
        assert agg["EVA-X_pass"] == 0.0
        assert agg["EVA-overall_pass"] == 0.0

    def test_missing_component_returns_none_for_pass(self):
        """Missing metric -> pass composite is None."""
        rm = make_record_metrics(
            {
                "task_completion": 1.0,
                # faithfulness missing
                "agent_speech_fidelity": 0.95,
            }
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_pass"] is None
        # EVA-overall_pass depends on EVA-A_pass which is None
        assert agg["EVA-overall_pass"] is None

    def test_missing_component_mean_uses_available(self):
        """Mean composites average only available metrics."""
        rm = make_record_metrics(
            {
                "task_completion": 1.0,
                # faithfulness missing
                "agent_speech_fidelity": 0.8,
            }
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_mean"] == pytest.approx((1.0 + 0.8) / 2)

    def test_no_components_available_mean_is_none(self):
        """Mean is None if no component metrics exist."""
        rm = make_record_metrics({})
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_mean"] is None
        assert agg["EVA-X_mean"] is None
        assert agg["EVA-overall_mean"] is None

    def test_error_metric_excluded(self):
        """Metrics with errors return None from get_score, so they're excluded."""
        rm = RecordMetrics(
            record_id="1.1.1",
            metrics={
                "task_completion": MetricScore(name="task_completion", score=0.0, error="LLM failed"),
                "faithfulness": MetricScore(name="faithfulness", score=0.5, normalized_score=0.5),
                "agent_speech_fidelity": MetricScore(name="agent_speech_fidelity", score=0.95, normalized_score=0.95),
            },
        )
        agg = compute_record_aggregates(rm)

        # task_completion has error -> None from get_score -> EVA-A_pass is None
        assert agg["EVA-A_pass"] is None
        # Mean only includes non-error scores
        assert agg["EVA-A_mean"] == pytest.approx((0.5 + 0.95) / 2)

    def test_skipped_component_excluded_from_pass(self):
        """Skipped component excluded from pass check.

        Remaining components still determine pass/fail.
        """
        rm = RecordMetrics(
            record_id="1.1.1",
            metrics={
                "task_completion": MetricScore(name="task_completion", score=1.0, normalized_score=1.0),
                "faithfulness": MetricScore(name="faithfulness", score=0.8, normalized_score=0.8),
                "agent_speech_fidelity": MetricScore(
                    name="agent_speech_fidelity",
                    score=None,
                    normalized_score=None,
                    skipped=True,
                ),
            },
        )
        agg = compute_record_aggregates(rm)

        # Skipped component is excluded; the two remaining components both pass
        assert agg["EVA-A_pass"] == 1.0

    def test_skipped_component_still_respects_other_failures(self):
        """A skipped component does not mask a real failure in another component."""
        rm = RecordMetrics(
            record_id="1.1.1",
            metrics={
                "task_completion": MetricScore(name="task_completion", score=0.5, normalized_score=0.5),
                "faithfulness": MetricScore(name="faithfulness", score=0.8, normalized_score=0.8),
                "agent_speech_fidelity": MetricScore(
                    name="agent_speech_fidelity", score=None, normalized_score=None, skipped=True
                ),
            },
        )
        agg = compute_record_aggregates(rm)

        # task_completion fails (0.5 != 1.0) -> EVA-A_pass is 0.0
        assert agg["EVA-A_pass"] == 0.0

    def test_all_components_skipped_pass_is_none(self):
        """If every component is skipped, the composite is None (nothing to evaluate)."""
        rm = RecordMetrics(
            record_id="1.1.1",
            metrics={
                "task_completion": MetricScore(name="task_completion", score=None, normalized_score=None, skipped=True),
                "faithfulness": MetricScore(name="faithfulness", score=None, normalized_score=None, skipped=True),
                "agent_speech_fidelity": MetricScore(
                    name="agent_speech_fidelity", score=None, normalized_score=None, skipped=True
                ),
            },
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_pass"] is None

    def test_agent_speech_fidelity_threshold_boundary(self):
        """agent_speech_fidelity uses > 0.9 (not >=), so 0.9 exactly fails."""
        rm = make_record_metrics(
            {
                "task_completion": 1.0,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.9,
            }
        )
        agg = compute_record_aggregates(rm)
        assert agg["EVA-A_pass"] == 0.0


class TestComputeRunLevelAggregates:
    def test_basic_run_level(self):
        """Basic run-level aggregation across 2 records."""
        r1 = make_record_metrics(
            {
                "task_completion": 1.0,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.8,
                "conciseness": 0.5,
            },
            record_id="1.1.1",
        )
        r1.aggregate_metrics = compute_record_aggregates(r1)

        r2 = make_record_metrics(
            {
                "task_completion": 0.5,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.3,
                "conciseness": 0.5,
            },
            record_id="1.1.2",
        )
        r2.aggregate_metrics = compute_record_aggregates(r2)

        result = compute_run_level_aggregates({"1.1.1": r1, "1.1.2": r2}, seed=42)

        # EVA-A_pass: r1=1.0, r2=0.0 -> mean=0.5
        assert result["EVA-A_pass"]["mean"] == 0.5
        assert result["EVA-A_pass"]["count"] == 2
        assert result["EVA-A_pass"]["success_rate"] == 0.5

        # EVA-X_pass: r1=1.0, r2=0.0 (turn_taking < 0.8) -> mean=0.5
        assert result["EVA-X_pass"]["mean"] == 0.5

    def test_mean_success_rate(self):
        """success_rate for mean composites counts records >= 0.5."""
        r1 = make_record_metrics(
            {
                "task_completion": 1.0,
                "faithfulness": 1.0,
                "agent_speech_fidelity": 1.0,
            },
            record_id="1",
        )
        r1.aggregate_metrics = compute_record_aggregates(r1)

        r2 = make_record_metrics(
            {
                "task_completion": 0.0,
                "faithfulness": 0.0,
                "agent_speech_fidelity": 0.0,
            },
            record_id="2",
        )
        r2.aggregate_metrics = compute_record_aggregates(r2)

        result = compute_run_level_aggregates({"1": r1, "2": r2}, seed=42)

        # EVA-A_mean: r1=1.0, r2=0.0 -> mean=0.5, success_rate=0.5 (1 of 2 >= 0.5)
        assert result["EVA-A_mean"]["mean"] == 0.5
        assert result["EVA-A_mean"]["success_rate"] == 0.5

    def test_empty_metrics(self):
        """No records -> empty result."""
        result = compute_run_level_aggregates({}, seed=42)
        assert result == {}

    def test_records_with_none_aggregates_excluded(self):
        """Records with None aggregate values are excluded from counts."""
        r1 = make_record_metrics({"task_completion": 1.0}, record_id="1")
        r1.aggregate_metrics = compute_record_aggregates(r1)
        # EVA-A_pass should be None (missing faithfulness, agent_speech_fidelity)
        assert r1.aggregate_metrics["EVA-A_pass"] is None

        result = compute_run_level_aggregates({"1": r1}, seed=42)

        # EVA-A_pass present but with None mean and none_count tracking
        assert result["EVA-A_pass"]["mean"] is None
        assert result["EVA-A_pass"]["count"] == 0
        assert result["EVA-A_pass"]["none_count"] == 1
        assert "success_rate" not in result["EVA-A_pass"]

    def test_pass_at_k_with_multi_trial(self):
        """pass@k computed for aggregate pass metrics when multi-trial."""
        all_metrics = {}
        for trial_idx in range(3):
            rm = make_record_metrics(
                {
                    "task_completion": 1.0 if trial_idx < 2 else 0.5,
                    "faithfulness": 0.5,
                    "agent_speech_fidelity": 0.95,
                    "conversation_progression": 0.5,
                    "turn_taking": 0.8,
                    "conciseness": 0.5,
                },
                record_id=f"1.1.1/trial_{trial_idx}",
            )
            rm.aggregate_metrics = compute_record_aggregates(rm)
            all_metrics[f"1.1.1/trial_{trial_idx}"] = rm

        result = compute_run_level_aggregates(all_metrics, num_draws=3, seed=42)

        assert "pass_k" in result
        eva_a = result["pass_k"]["EVA-A_pass"]
        assert eva_a["k"] == 3
        assert eva_a["count"] == 1
        # pass@1 = c/n = 2/3
        assert eva_a["pass_at_1"] == pytest.approx(2 / 3, abs=1e-4)
        # pass@k with k=n=3: 1.0 since c>=1
        assert eva_a["pass_at_k"] == 1.0
        # pass^k observed with k=n=3: 0.0 since c<k
        assert eva_a["pass_power_k_observed"] == 0.0
        # pass^k theoretical: (c/n)^k = (2/3)^3
        assert eva_a["pass_power_k_theoretical"] == pytest.approx((2 / 3) ** 3, abs=1e-4)

    def test_pass_at_k_excludes_record_with_none_trial(self):
        """Record with a None composite trial is excluded from pass@k entirely."""
        all_metrics = {}
        for trial_idx in range(3):
            scores = {
                "task_completion": 1.0,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.8,
                "conciseness": 0.5,
            }
            # Make trial_2 have an error in task_completion → EVA-A_pass becomes None
            if trial_idx == 2:
                scores.pop("faithfulness")  # Missing component → EVA-A_pass = None

            rm = make_record_metrics(scores, record_id=f"1.1.1/trial_{trial_idx}")
            rm.aggregate_metrics = compute_record_aggregates(rm)
            all_metrics[f"1.1.1/trial_{trial_idx}"] = rm

        # Verify trial 2 has None for EVA-A_pass
        assert all_metrics["1.1.1/trial_2"].aggregate_metrics["EVA-A_pass"] is None

        result = compute_run_level_aggregates(all_metrics, num_draws=3, seed=42)

        # Record should be excluded from pass_k since not all 3 trials are valid
        assert "pass_k" not in result or "EVA-A_pass" not in result.get("pass_k", {})


class TestScenarioGrouping:
    def test_per_metric_k1_record_equals_scenario(self):
        r1 = make_record_metrics({"task_completion": 1.0}, record_id="1.1.1")
        r2 = make_record_metrics({"task_completion": 0.5}, record_id="1.1.2")
        vals = scenario_means_for_metric({"1.1.1": r1, "1.1.2": r2}, "task_completion")
        np.testing.assert_allclose(sorted(vals), [0.5, 1.0])

    def test_per_metric_k3_collapses_trials(self):
        # Same scenario id "1.1.1", three trials with scores 0.0, 0.5, 1.0 → scenario mean 0.5
        r0 = make_record_metrics({"task_completion": 0.0}, record_id="1.1.1/trial_0")
        r1 = make_record_metrics({"task_completion": 0.5}, record_id="1.1.1/trial_1")
        r2 = make_record_metrics({"task_completion": 1.0}, record_id="1.1.1/trial_2")
        all_m = {"1.1.1/trial_0": r0, "1.1.1/trial_1": r1, "1.1.1/trial_2": r2}
        vals = scenario_means_for_metric(all_m, "task_completion")
        np.testing.assert_allclose(vals, [0.5])

    def test_per_metric_skips_errored_trials(self):
        # One scenario, two trials; one trial has the metric errored
        r0 = make_record_metrics({"task_completion": 1.0}, record_id="1.1.1/trial_0")
        r1 = RecordMetrics(
            record_id="1.1.1/trial_1",
            metrics={"task_completion": MetricScore(name="task_completion", score=0.0, error="boom")},
        )
        vals = scenario_means_for_metric({"1.1.1/trial_0": r0, "1.1.1/trial_1": r1}, "task_completion")
        np.testing.assert_allclose(vals, [1.0])  # mean over the 1 valid trial

    def test_per_metric_drops_all_none_scenarios(self):
        # Scenario with all trials errored is dropped from the bootstrap unit count.
        r0 = RecordMetrics(
            record_id="1.1.1/trial_0",
            metrics={"task_completion": MetricScore(name="task_completion", score=0.0, error="boom")},
        )
        r1 = RecordMetrics(
            record_id="1.1.1/trial_1",
            metrics={"task_completion": MetricScore(name="task_completion", score=0.0, error="boom")},
        )
        r2 = make_record_metrics({"task_completion": 0.5}, record_id="1.1.2/trial_0")
        all_m = {"1.1.1/trial_0": r0, "1.1.1/trial_1": r1, "1.1.2/trial_0": r2}
        vals = scenario_means_for_metric(all_m, "task_completion")
        np.testing.assert_allclose(vals, [0.5])

    def test_composite_k3_collapses_trials(self):
        # EVA-A_pass scenario value = mean over trials of per-trial 0/1
        comp = _composite_by_name("EVA-A_pass")
        r0 = make_record_metrics(
            {"task_completion": 1.0, "faithfulness": 0.5, "agent_speech_fidelity": 0.95},
            record_id="1.1.1/trial_0",
        )
        r0.aggregate_metrics = compute_record_aggregates(r0)
        r1 = make_record_metrics(
            {"task_completion": 0.0, "faithfulness": 0.5, "agent_speech_fidelity": 0.95},
            record_id="1.1.1/trial_1",
        )
        r1.aggregate_metrics = compute_record_aggregates(r1)
        all_m = {"1.1.1/trial_0": r0, "1.1.1/trial_1": r1}
        vals = _scenario_values_for_composite(all_m, comp)
        # trial 0 passes (1.0), trial 1 fails (0.0) → scenario mean 0.5
        np.testing.assert_allclose(vals, [0.5])

    def test_composite_empty_returns_empty_array(self):
        comp = _composite_by_name("EVA-A_pass")
        vals = _scenario_values_for_composite({}, comp)
        assert vals == []


class TestRunLevelCompositeCIs:
    def test_emits_ci_fields_for_all_composites(self):
        records = _make_clean_records(n=20, passing=10)
        result = compute_run_level_aggregates(records, seed=42)
        for comp_name in [
            "EVA-A_pass",
            "EVA-X_pass",
            "EVA-A_mean",
            "EVA-X_mean",
            "EVA-overall_mean",
            "EVA-overall_pass",
        ]:
            assert "mean_ci_lower" in result[comp_name], f"missing mean_ci_lower on {comp_name}"
            assert "mean_ci_upper" in result[comp_name], f"missing mean_ci_upper on {comp_name}"
            assert "mean_ci_n_scenarios" in result[comp_name], f"missing mean_ci_n_scenarios on {comp_name}"

    def test_ci_brackets_point_estimate(self):
        records = _make_clean_records(n=50, passing=25)
        result = compute_run_level_aggregates(records, seed=42)
        entry = result["EVA-A_pass"]
        assert entry["mean_ci_lower"] <= entry["mean"] <= entry["mean_ci_upper"]

    def test_n_scenarios_equals_count_for_k1(self):
        records = _make_clean_records(n=20, passing=10)
        result = compute_run_level_aggregates(records, seed=42)
        assert result["EVA-A_pass"]["mean_ci_n_scenarios"] == result["EVA-A_pass"]["count"]

    def test_empty_run_returns_empty_dict(self):
        result = compute_run_level_aggregates({}, seed=42)
        # The existing function already early-returns {} for empty input; CI
        # addition must not change this.
        assert result == {}

    def test_composite_with_no_valid_data_emits_null_ci(self):
        # A record where every component has an error → composite is None
        r = RecordMetrics(
            record_id="1.1.1",
            metrics={
                "task_completion": MetricScore(name="task_completion", score=0.0, error="boom"),
                "faithfulness": MetricScore(name="faithfulness", score=0.0, error="boom"),
                "agent_speech_fidelity": MetricScore(name="agent_speech_fidelity", score=0.0, error="boom"),
            },
        )
        r.aggregate_metrics = compute_record_aggregates(r)
        # Sanity: composite is None for this record
        assert r.aggregate_metrics["EVA-A_pass"] is None

        result = compute_run_level_aggregates({"1.1.1": r}, seed=42)
        entry = result["EVA-A_pass"]
        assert entry["mean_ci_lower"] is None
        assert entry["mean_ci_upper"] is None
        assert entry["mean_ci_n_scenarios"] == 0


class TestRunLevelPassKCIs:
    def _make_multi_trial_records(self, scenario_pass_pattern: list[tuple[int, int]]):
        """For each ``(n_scenarios, n_passing_trials_per_scenario)`` group, build records.

        Always uses k=3 trials per scenario.
        """
        records = {}
        sid = 0
        for n_scen, n_pass in scenario_pass_pattern:
            for _ in range(n_scen):
                sid += 1
                for trial in range(3):
                    is_pass = trial < n_pass
                    r = make_record_metrics(
                        {
                            "task_completion": 1.0 if is_pass else 0.0,
                            "faithfulness": 0.5,
                            "agent_speech_fidelity": 0.95,
                            "conversation_progression": 0.5,
                            "turn_taking": 0.8,
                            "conciseness": 0.5,
                        },
                        record_id=f"1.1.{sid}/trial_{trial}",
                    )
                    r.aggregate_metrics = compute_record_aggregates(r)
                    records[f"1.1.{sid}/trial_{trial}"] = r
        return records

    def test_pass_k_block_has_ci_fields(self):
        records = self._make_multi_trial_records([(10, 3), (10, 1), (10, 0)])
        result = compute_run_level_aggregates(records, num_draws=3, seed=42)
        block = result["pass_k"]["EVA-A_pass"]
        for stat in ["pass_at_1", "pass_at_k", "pass_power_k_observed"]:
            assert f"{stat}_ci_lower" in block, f"missing {stat}_ci_lower"
            assert f"{stat}_ci_upper" in block, f"missing {stat}_ci_upper"
        # pass_power_k_theoretical stays bare
        assert "pass_power_k_theoretical_ci_lower" not in block
        assert "pass_power_k_theoretical_ci_upper" not in block

    def test_pass_k_ci_brackets_point(self):
        records = self._make_multi_trial_records([(10, 3), (10, 1), (10, 0)])
        result = compute_run_level_aggregates(records, num_draws=3, seed=42)
        block = result["pass_k"]["EVA-A_pass"]
        assert block["pass_at_1_ci_lower"] <= block["pass_at_1"] <= block["pass_at_1_ci_upper"]
        assert block["pass_at_k_ci_lower"] <= block["pass_at_k"] <= block["pass_at_k_ci_upper"]
        assert (
            block["pass_power_k_observed_ci_lower"]
            <= block["pass_power_k_observed"]
            <= block["pass_power_k_observed_ci_upper"]
        )


class TestPerMetricCIs:
    def _records_with_metric(self, name: str, values: list[tuple[str, float | None]]):
        """Build a dict[record_id, RecordMetrics] from (record_id, value) pairs.

        ``None`` value means the metric is errored for that record.
        """
        out = {}
        for rid, v in values:
            if v is None:
                m = MetricScore(name=name, score=0.0, error="boom")
            else:
                m = MetricScore(name=name, score=v, normalized_score=v)
            out[rid] = RecordMetrics(record_id=rid, metrics={name: m})
        return out

    def test_per_metric_mean_ci_fields(self):
        records = self._records_with_metric(
            "task_completion",
            [(f"1.1.{i}", float(i) / 10) for i in range(20)],
        )
        agg = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=None, num_draws=1, seed=42
        )
        entry = agg["task_completion"]
        assert "mean_ci_lower" in entry
        assert "mean_ci_upper" in entry
        assert "mean_ci_n_scenarios" in entry
        assert entry["mean_ci_lower"] <= entry["mean"] <= entry["mean_ci_upper"]
        # n_scenarios == count for k=1
        assert entry["mean_ci_n_scenarios"] == entry["count"]

    def test_per_metric_no_valid_records_emits_null_ci(self):
        records = self._records_with_metric(
            "task_completion",
            [("1.1.1", None), ("1.1.2", None)],
        )
        agg = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=None, num_draws=1, seed=42
        )
        entry = agg["task_completion"]
        assert entry["mean_ci_lower"] is None
        assert entry["mean_ci_upper"] is None
        assert entry["mean_ci_n_scenarios"] == 0

    def test_per_metric_pass_k_ci_fields(self):
        # Build per-scenario PassAtKResult fixtures and confirm pass_k CI fields appear.
        records = {}
        for sid in range(10):
            for trial in range(3):
                m = MetricScore(
                    name="task_completion", score=1.0 if trial < 2 else 0.0, normalized_score=1.0 if trial < 2 else 0.0
                )
                records[f"1.1.{sid}/trial_{trial}"] = RecordMetrics(
                    record_id=f"1.1.{sid}/trial_{trial}",
                    metrics={"task_completion": m},
                )
        pass_at_k_results = {
            f"1.1.{sid}": {
                "task_completion": PassAtKResult(
                    metric_name="task_completion",
                    n=3,
                    k=3,
                    c=2,
                    pass_at_k=1.0,
                    pass_power_k=0.0,
                    threshold=0.5,
                )
            }
            for sid in range(10)
        }
        agg = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=pass_at_k_results, num_draws=3, seed=42
        )
        block = agg["task_completion"]["pass_k"]
        for stat in ["pass_at_1", "pass_at_k", "pass_power_k_observed"]:
            assert f"{stat}_ci_lower" in block
            assert f"{stat}_ci_upper" in block


class TestRunSeedIntegration:
    def test_within_run_byte_identical(self):
        records = _make_clean_records(n=20, passing=10)
        seed = run_seed("2026-04-16_18-55-44.848147_gpt-realtime-1.5")
        a = compute_run_level_aggregates(records, seed=seed)
        b = compute_run_level_aggregates(records, seed=seed)
        assert a == b

    def test_across_run_independence(self):
        records = _make_clean_records(n=20, passing=10)
        # Seed strings chosen empirically: the bimodal n=20 fixture gives a low-variance
        # bootstrap distribution where many seed pairs land on identical percentile bounds.
        # The "x"/"y" pair produces differing CI bounds for both EVA-A_pass and EVA-A_mean.
        seed_a = run_seed("x")
        seed_b = run_seed("y")
        a = compute_run_level_aggregates(records, seed=seed_a)
        b = compute_run_level_aggregates(records, seed=seed_b)
        # Point estimates are identical (same data); CI bounds differ (different MC noise).
        for comp_name in ["EVA-A_pass", "EVA-A_mean"]:
            assert a[comp_name]["mean"] == b[comp_name]["mean"]
            # At least one of (lower, upper) must differ across runs.
            assert (
                a[comp_name]["mean_ci_lower"] != b[comp_name]["mean_ci_lower"]
                or a[comp_name]["mean_ci_upper"] != b[comp_name]["mean_ci_upper"]
            )

    def test_per_metric_seed_propagation(self):
        # The seed kwarg added in Task 5 to _build_per_metric_aggregates must actually
        # change the CI bounds; same data + same seed must be deterministic.
        records = {}
        for i in range(20):
            value = float(i) / 20.0
            m = MetricScore(name="task_completion", score=value, normalized_score=value)
            records[f"1.1.{i}"] = RecordMetrics(record_id=f"1.1.{i}", metrics={"task_completion": m})

        seed_a = run_seed("run-a")
        seed_b = run_seed("run-b")

        agg_a1 = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=None, num_draws=1, seed=seed_a
        )
        agg_a2 = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=None, num_draws=1, seed=seed_a
        )
        agg_b = MetricsRunner._build_per_metric_aggregates(
            records, ["task_completion"], pass_at_k_results=None, num_draws=1, seed=seed_b
        )

        # Same seed → byte-identical
        assert agg_a1["task_completion"] == agg_a2["task_completion"]
        # Different seed → at least one bound differs. The n=20 continuous-value fixture
        # produces enough bootstrap variance for bounds to differ across seeds.
        entry_a = agg_a1["task_completion"]
        entry_b = agg_b["task_completion"]
        assert entry_a["mean"] == entry_b["mean"]
        assert (
            entry_a["mean_ci_lower"] != entry_b["mean_ci_lower"] or entry_a["mean_ci_upper"] != entry_b["mean_ci_upper"]
        )
