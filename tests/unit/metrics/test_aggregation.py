"""Unit tests for EVA composite metric aggregation."""

import pytest

from eva.metrics.aggregation import (
    _check_threshold,
    compute_record_aggregates,
    compute_run_level_aggregates,
)
from eva.models.results import MetricScore, RecordMetrics

from .conftest import make_record_metrics


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
                "turn_taking": 0.5,
                "conciseness": 0.5,
            }
        )
        agg = compute_record_aggregates(rm)

        assert agg["EVA-A_pass"] == 1.0
        assert agg["EVA-X_pass"] == 1.0
        assert agg["EVA-overall_pass"] == 1.0
        assert agg["EVA-A_mean"] == pytest.approx((1.0 + 0.5 + 0.95) / 3)
        assert agg["EVA-X_mean"] == pytest.approx((0.5 + 0.5 + 0.5) / 3)
        assert agg["EVA-overall_mean"] == pytest.approx((1.0 + 0.5 + 0.95 + 0.5 + 0.5 + 0.5) / 6)

    def test_eva_a_fails(self):
        """task_completion < 1.0 causes EVA-A_pass to fail."""
        rm = make_record_metrics(
            {
                "task_completion": 0.5,
                "faithfulness": 0.5,
                "agent_speech_fidelity": 0.95,
                "conversation_progression": 0.5,
                "turn_taking": 0.5,
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
                "turn_taking": 0.5,
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
                "turn_taking": 0.5,
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

        result = compute_run_level_aggregates({"1.1.1": r1, "1.1.2": r2})

        # EVA-A_pass: r1=1.0, r2=0.0 -> mean=0.5
        assert result["EVA-A_pass"]["mean"] == 0.5
        assert result["EVA-A_pass"]["count"] == 2
        assert result["EVA-A_pass"]["success_rate"] == 0.5

        # EVA-X_pass: r1=1.0, r2=0.0 (turn_taking < 0.5) -> mean=0.5
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

        result = compute_run_level_aggregates({"1": r1, "2": r2})

        # EVA-A_mean: r1=1.0, r2=0.0 -> mean=0.5, success_rate=0.5 (1 of 2 >= 0.5)
        assert result["EVA-A_mean"]["mean"] == 0.5
        assert result["EVA-A_mean"]["success_rate"] == 0.5

    def test_empty_metrics(self):
        """No records -> empty result."""
        result = compute_run_level_aggregates({})
        assert result == {}

    def test_records_with_none_aggregates_excluded(self):
        """Records with None aggregate values are excluded from counts."""
        r1 = make_record_metrics({"task_completion": 1.0}, record_id="1")
        r1.aggregate_metrics = compute_record_aggregates(r1)
        # EVA-A_pass should be None (missing faithfulness, agent_speech_fidelity)
        assert r1.aggregate_metrics["EVA-A_pass"] is None

        result = compute_run_level_aggregates({"1": r1})

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
                    "turn_taking": 0.5,
                    "conciseness": 0.5,
                },
                record_id=f"1.1.1/trial_{trial_idx}",
            )
            rm.aggregate_metrics = compute_record_aggregates(rm)
            all_metrics[f"1.1.1/trial_{trial_idx}"] = rm

        result = compute_run_level_aggregates(all_metrics, num_draws=3)

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
                "turn_taking": 0.5,
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

        result = compute_run_level_aggregates(all_metrics, num_draws=3)

        # Record should be excluded from pass_k since not all 3 trials are valid
        assert "pass_k" not in result or "EVA-A_pass" not in result.get("pass_k", {})
