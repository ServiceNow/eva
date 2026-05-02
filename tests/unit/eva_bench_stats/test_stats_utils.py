"""Unit tests for analysis/eva-bench-stats/stats_utils.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "eva-bench-stats"))

from stats_utils import bootstrap_ci, bootstrap_resample  # noqa: E402


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
