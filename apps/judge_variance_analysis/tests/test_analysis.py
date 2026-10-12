import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_scores_df(data: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(data)


def make_agg_df(data: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(data)


class TestJudgeVariance:
    def test_std_zero_when_identical(self):
        from judge_variance_analysis.analysis import compute_judge_variance

        df = make_scores_df(
            [
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "iteration": 1,
                    "metric": "faithfulness",
                    "normalized_score": 0.5,
                },
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "iteration": 2,
                    "metric": "faithfulness",
                    "normalized_score": 0.5,
                },
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "iteration": 3,
                    "metric": "faithfulness",
                    "normalized_score": 0.5,
                },
            ]
        )
        result = compute_judge_variance(df, ["faithfulness"])
        assert result.iloc[0]["std"] == pytest.approx(0.0)
        assert result.iloc[0]["range"] == pytest.approx(0.0)
        assert result.iloc[0]["score_changed"] is False

    def test_std_nonzero_when_different(self):
        from judge_variance_analysis.analysis import compute_judge_variance

        df = make_scores_df(
            [
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "iteration": 1,
                    "metric": "faithfulness",
                    "normalized_score": 0.0,
                },
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "iteration": 2,
                    "metric": "faithfulness",
                    "normalized_score": 0.5,
                },
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "iteration": 3,
                    "metric": "faithfulness",
                    "normalized_score": 1.0,
                },
            ]
        )
        result = compute_judge_variance(df, ["faithfulness"])
        assert result.iloc[0]["std"] == pytest.approx(np.std([0.0, 0.5, 1.0]))
        assert result.iloc[0]["range"] == pytest.approx(1.0)
        assert result.iloc[0]["score_changed"] is True


class TestTrajectoryVariance:
    def test_std_across_trials(self):
        from judge_variance_analysis.analysis import compute_trajectory_variance

        # Same scores across iterations, but different across trials
        rows = []
        for trial, score in enumerate([0.2, 0.5, 0.8]):
            for iteration in [1, 2, 3]:
                rows.append(
                    {
                        "run_id": "r1",
                        "run_label": "R1",
                        "record_id": "1.1",
                        "trial": trial,
                        "iteration": iteration,
                        "metric": "faithfulness",
                        "normalized_score": score,
                    }
                )
        df = make_scores_df(rows)
        result = compute_trajectory_variance(df, ["faithfulness"])
        # Mean scores per trial: 0.2, 0.5, 0.8 → std ≈ 0.245
        expected_std = np.std([0.2, 0.5, 0.8])
        assert result.iloc[0]["std"] == pytest.approx(expected_std, abs=1e-6)


class TestCompositeStability:
    def test_stable_when_all_pass(self):
        from judge_variance_analysis.analysis import compute_composite_stability

        # All trials pass in all iterations → pass@1=1.0, pass@k=1.0, pass^k=1.0
        rows = []
        for iteration in [1, 2, 3]:
            for record_id in ["1.1", "1.2"]:
                for trial in [0, 1, 2]:
                    rows.append(
                        {
                            "run_id": "r1",
                            "run_label": "R1",
                            "record_id": record_id,
                            "trial": trial,
                            "iteration": iteration,
                            "EVA-overall_pass": 1.0,
                            "EVA-A_mean": 1.0,
                            "EVA-X_mean": 1.0,
                        }
                    )
        df = make_agg_df(rows)
        result = compute_composite_stability(df)
        assert len(result) == 3  # 3 iterations
        assert (result["EVA-overall_pass_at_1"] == 1.0).all()
        assert (result["EVA-overall_pass_at_k"] == 1.0).all()

    def test_pass_at_k_1_if_any_trial_passes(self):
        from judge_variance_analysis.analysis import compute_composite_stability

        # For record_id 1.1: only trial 0 passes (c=1, n=3)
        rows = []
        for iteration in [1, 2, 3]:
            for trial, val in enumerate([1.0, 0.0, 0.0]):
                rows.append(
                    {
                        "run_id": "r1",
                        "run_label": "R1",
                        "record_id": "1.1",
                        "trial": trial,
                        "iteration": iteration,
                        "EVA-overall_pass": val,
                        "EVA-A_mean": 0.5,
                        "EVA-X_mean": 0.5,
                    }
                )
        df = make_agg_df(rows)
        result = compute_composite_stability(df)
        # pass@k = 1.0 (at least one trial passed)
        assert (result["EVA-overall_pass_at_k"] == 1.0).all()
        # pass@1 = 1/3
        assert result.iloc[0]["EVA-overall_pass_at_1"] == pytest.approx(1 / 3)
        # pass^k = (1/3)^3
        assert result.iloc[0]["EVA-overall_pass_power_k"] == pytest.approx((1 / 3) ** 3)


class TestBorderlineScenarios:
    def test_no_borderline_when_stable(self):
        from judge_variance_analysis.analysis import find_borderline_scenarios

        df = make_scores_df(
            [
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "metric": "faithfulness",
                    "std": 0.0,
                    "range": 0.0,
                    "score_changed": False,
                },
            ]
        )
        result = find_borderline_scenarios(df)
        assert len(result) == 0

    def test_finds_borderline_when_score_changed(self):
        from judge_variance_analysis.analysis import find_borderline_scenarios

        df = make_scores_df(
            [
                {
                    "run_id": "r1",
                    "run_label": "R1",
                    "record_id": "1.1",
                    "trial": 0,
                    "metric": "faithfulness",
                    "std": 0.3,
                    "range": 0.5,
                    "score_changed": True,
                },
            ]
        )
        result = find_borderline_scenarios(df)
        assert len(result) == 1


class TestOnewayIcc:
    """Tests for the private _oneway_icc helper (one-way ANOVA ICC)."""

    def _call(self, groups):
        from judge_variance_analysis.analysis import _oneway_icc

        return _oneway_icc(groups)

    def test_icc_one_when_perfect_scenario_differentiation(self):
        # All scores identical within each group, different across groups
        # → zero within-group variance, non-zero between-group variance → ICC = 1
        groups = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 1.0, 1.0]),
        ]
        r = self._call(groups)
        assert r["icc"] == pytest.approx(1.0, abs=1e-6)

    def test_icc_zero_when_no_scenario_differentiation(self):
        # All groups have the same mean → zero between-group variance → ICC = 0
        groups = [
            np.array([0.3, 0.5, 0.7]),
            np.array([0.3, 0.5, 0.7]),
            np.array([0.3, 0.5, 0.7]),
        ]
        r = self._call(groups)
        assert r["icc"] == pytest.approx(0.0, abs=1e-6)

    def test_ci_bounds_in_unit_interval(self):
        groups = [np.array([0.2, 0.4, 0.6]), np.array([0.5, 0.7, 0.9])]
        r = self._call(groups)
        assert 0.0 <= r["ci_lower"] <= r["icc"] <= r["ci_upper"] <= 1.0

    def test_sigma2_nonnegative(self):
        # When between-group variance is smaller than within, sigma2_scenario is clipped to 0
        groups = [
            np.array([0.1, 0.9]),  # high within-group spread
            np.array([0.1, 0.9]),  # same for all groups → no between-group signal
        ]
        r = self._call(groups)
        assert r["sigma2_scenario"] >= 0.0
        assert r["sigma2_residual"] >= 0.0

    def test_returns_expected_keys(self):
        groups = [np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.4, 0.6])]
        r = self._call(groups)
        for key in (
            "icc",
            "ci_lower",
            "ci_upper",
            "sigma2_scenario",
            "sigma2_residual",
            "ms_between",
            "ms_within",
            "f_stat",
            "p_value",
        ):
            assert key in r, f"missing key: {key}"


def _make_icc_scores(scenarios, models, trials=3, iterations=3, metric="faithfulness", score_fn=None):
    """Build a minimal scores DataFrame for ICC testing.

    score_fn(scenario_idx, model_idx, trial_idx) → float.
    Defaults to 0.5 everywhere.
    """

    def _default_score(s, m, t):
        return 0.5

    if score_fn is None:
        score_fn = _default_score
    rows = []
    for si, scenario in enumerate(scenarios):
        for mi, model in enumerate(models):
            for ti, trial in enumerate(range(trials)):
                for iteration in range(1, iterations + 1):
                    rows.append(
                        {
                            "run_id": model,
                            "run_label": model.upper(),
                            "record_id": scenario,
                            "trial": trial,
                            "iteration": iteration,
                            "metric": metric,
                            "normalized_score": score_fn(si, mi, ti),
                        }
                    )
    return pd.DataFrame(rows)


class TestComputeIcc:
    def test_per_model_shape(self):
        from judge_variance_analysis.analysis import compute_icc

        scenarios = [f"s{i}" for i in range(5)]
        models = ["r1", "r2"]
        df = _make_icc_scores(scenarios, models)
        result = compute_icc(df, ["faithfulness"])
        assert "per_model" in result
        pm = result["per_model"]
        assert len(pm) == 2  # one row per model
        assert set(pm.columns) >= {
            "run_id",
            "run_label",
            "metric",
            "icc",
            "ci_lower",
            "ci_upper",
            "sigma2_scenario",
            "sigma2_residual",
            "ms_between",
            "ms_within",
            "n_scenarios",
            "n_trials",
            "f_stat",
            "p_value",
        }

    def test_per_model_icc_one_when_only_scenario_varies(self):
        from judge_variance_analysis.analysis import compute_icc

        # score = scenario index → perfect between-scenario differentiation
        scenarios = [f"s{i}" for i in range(6)]
        df = _make_icc_scores(
            scenarios,
            ["r1"],
            score_fn=lambda s, m, t: s / 5.0,  # varies only with scenario
        )
        result = compute_icc(df, ["faithfulness"])
        row = result["per_model"].iloc[0]
        assert row["icc"] == pytest.approx(1.0, abs=1e-6)

    def test_pooled_centered_shape(self):
        from judge_variance_analysis.analysis import compute_icc

        scenarios = [f"s{i}" for i in range(5)]
        df = _make_icc_scores(scenarios, ["r1", "r2"])
        result = compute_icc(df, ["faithfulness"])
        assert "pooled_centered" in result
        pc = result["pooled_centered"]
        assert len(pc) == 1  # one row per metric
        assert set(pc.columns) >= {
            "metric",
            "icc",
            "ci_lower",
            "ci_upper",
            "sigma2_scenario",
            "sigma2_residual",
            "n_scenarios",
            "n_models",
            "n_trials",
        }

    def test_pooled_centered_removes_model_offset(self):
        from judge_variance_analysis.analysis import compute_icc

        # Model 0 has scores offset by 0.5 vs model 1, but same scenario pattern
        # After centering, ICC should match per-model ICC (scenario pattern is same)
        scenarios = [f"s{i}" for i in range(6)]
        df = _make_icc_scores(
            scenarios,
            ["r1", "r2"],
            score_fn=lambda s, m, t: s / 5.0 + m * 0.5,
        )
        result = compute_icc(df, ["faithfulness"])
        pc_icc = result["pooled_centered"].iloc[0]["icc"]
        pm_icc0 = result["per_model"].iloc[0]["icc"]
        pm_icc1 = result["per_model"].iloc[1]["icc"]
        # Both models have identical scenario pattern → per-model ICC = 1.0
        # After centering, pooled ICC should also be 1.0
        assert pc_icc == pytest.approx(1.0, abs=1e-6)
        assert pm_icc0 == pytest.approx(1.0, abs=1e-6)
        assert pm_icc1 == pytest.approx(1.0, abs=1e-6)


class TestTwowayIcc:
    def _call(self, Y):
        """Y shape: (n_scenarios, n_models, n_trials)."""
        from judge_variance_analysis.analysis import _twoway_icc

        return _twoway_icc(Y)

    def test_ss_partition(self):
        # SS_total must equal SS_scenario + SS_model + SS_interaction + SS_residual
        rng = np.random.default_rng(0)
        Y = rng.uniform(0, 1, (8, 3, 3))
        r = self._call(Y)
        ss_total = r["ss_scenario"] + r["ss_model"] + r["ss_interaction"] + r["ss_residual"]
        expected = float(((Y - Y.mean()) ** 2).sum())
        assert ss_total == pytest.approx(expected, rel=1e-6)

    def test_variance_components_nonnegative(self):
        rng = np.random.default_rng(1)
        Y = rng.uniform(0, 1, (5, 2, 3))
        r = self._call(Y)
        for key in ("sigma2_scenario", "sigma2_model", "sigma2_interaction", "sigma2_residual"):
            assert r[key] >= 0.0, f"{key} is negative"

    def test_icc_one_when_only_scenario_varies(self):
        # Y[i, j, k] = i / (n_scenarios - 1) → scenario effect only, no noise
        n_s, n_m, n_t = 6, 3, 3
        Y = np.zeros((n_s, n_m, n_t))
        for i in range(n_s):
            Y[i, :, :] = i / (n_s - 1)
        r = self._call(Y)
        # ICC_scenario should be 1 (all variance from scenario)
        assert r["icc_scenario"] == pytest.approx(1.0, abs=1e-6)
        assert r["icc_model"] == pytest.approx(0.0, abs=1e-6)

    def test_pooled_twoway_shape(self):
        from judge_variance_analysis.analysis import compute_icc

        scenarios = [f"s{i}" for i in range(6)]
        df = _make_icc_scores(scenarios, ["r1", "r2", "r3"])
        result = compute_icc(df, ["faithfulness"])
        tw = result["pooled_twoway"]
        assert len(tw) == 1
        assert set(tw.columns) >= {
            "metric",
            "icc_scenario",
            "icc_model",
            "ci_lower_scenario",
            "ci_upper_scenario",
            "ci_lower_model",
            "ci_upper_model",
            "sigma2_scenario",
            "sigma2_model",
            "sigma2_interaction",
            "sigma2_residual",
            "sigma2_total",
            "ms_scenario",
            "ms_model",
            "ms_interaction",
            "ms_residual",
            "f_scenario",
            "p_scenario",
            "f_model",
            "p_model",
            "f_interaction",
            "p_interaction",
            "n_scenarios",
            "n_models",
            "n_trials",
        }
