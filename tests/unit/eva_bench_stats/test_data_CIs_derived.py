from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from data_CIs import (
    DERIVED_PASS_METRICS,
    compute_derived_pass_metrics,
    check_completeness,
)


def _trials(alias: str = "sys-a") -> pd.DataFrame:
    """Two scenarios for sys-a, k=5 trials each, EVA-A_pass and EVA-X_pass."""
    rows = []
    # scenario 1: EVA-A_pass = [1,1,1,1,1]; EVA-X_pass = [0,0,0,0,0]
    for t, av, xv in zip(range(5), [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]):
        rows.append({"system_alias": alias, "domain": "itsm", "scenario_id": 1,
                     "trial": t, "metric": "EVA-A_pass", "value": float(av),
                     "perturbation_category": "clean"})
        rows.append({"system_alias": alias, "domain": "itsm", "scenario_id": 1,
                     "trial": t, "metric": "EVA-X_pass", "value": float(xv),
                     "perturbation_category": "clean"})
    # scenario 2: EVA-A_pass = [1,0,0,0,0]; EVA-X_pass = [1,1,1,1,1]
    for t, av, xv in zip(range(5), [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]):
        rows.append({"system_alias": alias, "domain": "itsm", "scenario_id": 2,
                     "trial": t, "metric": "EVA-A_pass", "value": float(av),
                     "perturbation_category": "clean"})
        rows.append({"system_alias": alias, "domain": "itsm", "scenario_id": 2,
                     "trial": t, "metric": "EVA-X_pass", "value": float(xv),
                     "perturbation_category": "clean"})
    return pd.DataFrame(rows)


def test_pass_at_k_is_max_over_trials() -> None:
    df = _trials()
    out = compute_derived_pass_metrics(df, "sys-a",
                                       ["EVA-A_pass_at_k"], expected_k=5)
    sc1 = out[(out["scenario_id"] == 1) & (out["metric"] == "EVA-A_pass_at_k")].iloc[0]
    sc2 = out[(out["scenario_id"] == 2) & (out["metric"] == "EVA-A_pass_at_k")].iloc[0]
    assert sc1["scenario_mean"] == 1.0   # max([1,1,1,1,1])
    assert sc2["scenario_mean"] == 1.0   # max([1,0,0,0,0])


def test_pass_power_k_is_mean_pow_k() -> None:
    df = _trials()
    out = compute_derived_pass_metrics(df, "sys-a",
                                       ["EVA-A_pass_power_k"], expected_k=5)
    sc1 = out[(out["scenario_id"] == 1) & (out["metric"] == "EVA-A_pass_power_k")].iloc[0]
    sc2 = out[(out["scenario_id"] == 2) & (out["metric"] == "EVA-A_pass_power_k")].iloc[0]
    assert sc1["scenario_mean"] == pytest.approx(1.0 ** 5)
    assert sc2["scenario_mean"] == pytest.approx((1.0 / 5.0) ** 5)


def test_eva_x_variants_independent_from_a() -> None:
    df = _trials()
    out = compute_derived_pass_metrics(df, "sys-a",
                                       ["EVA-X_pass_at_k", "EVA-X_pass_power_k"],
                                       expected_k=5)
    x_at_k = out[(out["scenario_id"] == 1) & (out["metric"] == "EVA-X_pass_at_k")].iloc[0]
    x_pow_k = out[(out["scenario_id"] == 2) & (out["metric"] == "EVA-X_pass_power_k")].iloc[0]
    assert x_at_k["scenario_mean"] == 0.0   # max([0,0,0,0,0])
    assert x_pow_k["scenario_mean"] == pytest.approx(1.0)  # mean([1,1,1,1,1])^5


def test_compute_derived_no_overlap_with_other_aliases() -> None:
    df = pd.concat([_trials("sys-a"), _trials("sys-b")], ignore_index=True)
    out = compute_derived_pass_metrics(df, "sys-a",
                                       ["EVA-A_pass_at_k"], expected_k=5)
    assert (out["system_alias"] == "sys-a").all()


def test_check_completeness_ignores_derived_metrics() -> None:
    df = _trials()
    metrics = ["EVA-A_pass", "EVA-X_pass",
               "EVA-A_pass_at_k", "EVA-A_pass_power_k",
               "EVA-X_pass_at_k", "EVA-X_pass_power_k"]
    expected_scenarios = {"itsm": 2}
    complete, rows = check_completeness(df, "sys-a", expected_scenarios,
                                        expected_k=5, metrics=metrics)
    # Derived metrics should NOT cause "n_metrics_missing" to be nonzero
    assert complete, rows
    assert all(r["n_metrics_missing"] == 0 for r in rows)
