"""Unit tests for analysis/eva-bench-stats/stats_utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "eva-bench-stats"))

from stats_utils import bootstrap_ci, bootstrap_resample, bootstrap_slope_ci  # noqa: E402


def test_bootstrap_resample_shape_and_determinism():
    values = np.array([0.0, 0.5, 1.0, 0.25, 0.75])
    a = bootstrap_resample(values, n_boot=100, seed=42)
    b = bootstrap_resample(values, n_boot=100, seed=42)
    assert a.shape == (100,)
    np.testing.assert_array_equal(a, b)


def test_bootstrap_resample_different_seeds_differ():
    values = np.array([0.0, 0.5, 1.0])
    a = bootstrap_resample(values, n_boot=100, seed=1)
    b = bootstrap_resample(values, n_boot=100, seed=2)
    assert not np.array_equal(a, b)


def test_bootstrap_resample_constant_input_constant_output():
    values = np.full(10, 0.7)
    boot = bootstrap_resample(values, n_boot=50, seed=0)
    np.testing.assert_allclose(boot, 0.7)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=100)
    lo, hi = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)
    assert lo < values.mean() < hi
    assert hi - lo < 0.1


def test_bootstrap_ci_alpha_widens_to_narrows():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=100)
    lo90, hi90 = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.10)
    lo95, hi95 = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)
    assert (hi95 - lo95) > (hi90 - lo90)


def test_bootstrap_slope_ci_returns_floats():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 11)
    y = 0.5 - 0.3 * x + rng.normal(0, 0.05, 11)
    lo, hi = bootstrap_slope_ci(x, y, quantile=0.75, n_boot=100, seed=0)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bootstrap_slope_ci_lower_le_upper():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 11)
    y = 0.8 - 0.5 * x + rng.normal(0, 0.03, 11)
    lo, hi = bootstrap_slope_ci(x, y, quantile=0.75, n_boot=200, seed=1)
    assert lo <= hi


def test_bootstrap_slope_ci_negative_slope():
    """For strongly negatively-related data, CI upper bound should be negative."""
    rng = np.random.default_rng(2)
    x = np.linspace(0.1, 0.9, 11)
    y = 1.0 - 0.9 * x + rng.normal(0, 0.01, 11)
    lo, hi = bootstrap_slope_ci(x, y, quantile=0.75, n_boot=500, seed=2)
    assert hi < 0, f"Expected negative CI upper bound for strong negative slope, got hi={hi}"


def test_permutation_test_all_zero_deltas():
    from stats_utils import permutation_test

    p = permutation_test(np.zeros(20), seed=0)
    assert p == 1.0


def test_permutation_test_large_positive_effect_low_p():
    from stats_utils import permutation_test

    deltas = np.full(50, 0.3)
    p = permutation_test(deltas, n_perm=1000, seed=42)
    assert p < 0.05


def test_permutation_test_deterministic():
    from stats_utils import permutation_test

    rng = np.random.default_rng(0)
    deltas = rng.normal(0.1, 0.2, 30)
    p1 = permutation_test(deltas, seed=7)
    p2 = permutation_test(deltas, seed=7)
    assert p1 == p2


def test_permutation_test_p_in_unit_interval():
    from stats_utils import permutation_test

    rng = np.random.default_rng(1)
    deltas = rng.normal(0, 0.1, 20)
    p = permutation_test(deltas, n_perm=200, seed=1)
    assert 0.0 <= p <= 1.0
