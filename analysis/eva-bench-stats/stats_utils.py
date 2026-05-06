"""Shared statistical primitives for eva-bench-stats analyses.

Pure NumPy. No file I/O, no plotting. Consumed by stats_perturbations and stats_CIs.
"""

from __future__ import annotations

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


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


def bootstrap_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrapped CI on the quantile regression slope.

    Resamples rows (models) with replacement and refits QuantReg on each
    resample. CI is the (alpha/2, 1-alpha/2) percentile of the slope
    distribution across successful resamples.

    Bootstrapped CIs are preferred over analytical SEs for quantile regression
    at small n: analytical SEs assume n >> n_models (here n_models = 11), and
    the bootstrap distribution of the slope is often non-normal at small n.

    Returns (nan, nan) if fewer than 10 resamples converge — signals the fit
    is unreliable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    slopes: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_boot, y_boot = x[idx], y[idx]
        if np.ptp(x_boot) < 1e-10:
            continue
        X_boot = np.column_stack([np.ones(n), x_boot])
        try:
            result = QuantReg(y_boot, X_boot).fit(q=quantile, max_iter=2000)
            slopes.append(float(result.params[1]))
        except Exception:
            continue

    if len(slopes) < 10:
        return float("nan"), float("nan")

    lower = float(np.percentile(slopes, 100 * alpha / 2))
    upper = float(np.percentile(slopes, 100 * (1 - alpha / 2)))
    return lower, upper


def permutation_test(
    deltas: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip permutation test.

    For each permutation, independently flip the sign of each delta with p=0.5,
    compute the mean. P-value = fraction of permutations where |permuted mean|
    >= |observed mean|.

    Args:
        deltas: 1-D array of scenario-level deltas.
        n_perm: Number of permutations.
        seed: RNG seed for reproducibility.

    Returns:
        Two-sided p-value in [0, 1].
    """
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    observed = np.mean(deltas)

    if observed == 0.0 and np.all(deltas == 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    permuted_means = (signs * deltas).mean(axis=1)

    p = np.mean(np.abs(permuted_means) >= np.abs(observed))
    return float(p)
