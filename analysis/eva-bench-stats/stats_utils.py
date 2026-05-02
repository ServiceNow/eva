"""Shared statistical primitives for eva-bench-stats analyses.

Pure NumPy. No file I/O, no plotting. Consumed by stats_perturbations and stats_CIs.
"""
from __future__ import annotations

import numpy as np


def bootstrap_resample(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Return shape (n_boot,) array of bootstrap-resample means.

    Resamples `values` with replacement `n_boot` times and returns the mean
    of each resample. Callers needing a percentile CI should use bootstrap_ci.

    Args:
        values: 1-D array of observations (e.g. scenario-level scores).
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        1-D float array of length n_boot containing the resample means.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.zeros(n_boot, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    return values[indices].mean(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval on the mean.

    Args:
        values: 1-D array of observations.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        alpha: Significance level; CI covers 1 - alpha probability.

    Returns:
        (lower, upper) CI bounds at the alpha/2 and 1-alpha/2 percentiles.
    """
    boot_means = bootstrap_resample(values, n_boot=n_boot, seed=seed)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper
