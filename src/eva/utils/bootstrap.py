"""Percentile bootstrap primitives for sample-mean confidence intervals.

This module is pure: numpy in, numpy/floats out. It has no eva imports and
is safe to use from anywhere in the package.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

N_BOOT = 2000
ALPHA = 0.05


def run_seed(run_id: str) -> int:
    """Stable, run-dependent seed derived from the run directory name.

    Uses ``hashlib.sha256`` rather than Python's built-in ``hash()`` because the
    latter is salted per interpreter process — re-invoking ``eva metrics`` on the
    same run would otherwise yield slightly different CI bounds. SHA-based hashing
    is byte-stable across processes.
    """
    h = hashlib.sha256(run_id.encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def bootstrap_resample(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Return ``n_boot`` resampled means of ``values``.

    Returns a zero-length array for empty input.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    *,
    seed: int,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """95% bootstrap CI on the mean (default alpha=0.05).

    ``seed`` is keyword-only and required: callers must supply a deliberate
    seed (typically from ``run_seed(run_dir.name)``) so behavior is deterministic.

    Returns ``(lower, upper)``; ``(nan, nan)`` if the input is empty.
    """
    boot = bootstrap_resample(values, n_boot=n_boot, seed=seed)
    if len(boot) == 0:
        return float("nan"), float("nan")
    lower = float(np.percentile(boot, 100 * alpha / 2))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_ci_fields(
    samples: dict[str, Sequence[float]],
    *,
    seed: int,
    decimals: int = 4,
) -> dict[str, float]:
    """Return ``{name}_ci_lower`` / ``{name}_ci_upper`` for each ``(name, sample)`` pair."""
    out: dict[str, float] = {}
    for name, sample in samples.items():
        lower, upper = bootstrap_ci(sample, seed=seed)
        out[f"{name}_ci_lower"] = round(lower, decimals)
        out[f"{name}_ci_upper"] = round(upper, decimals)
    return out


def mean_ci_fields(
    scenario_values: Sequence[float],
    *,
    seed: int,
    decimals: int = 4,
) -> dict[str, Any]:
    """Return ``mean_ci_lower`` / ``mean_ci_upper`` / ``mean_ci_n_scenarios``.

    Empty ``scenario_values`` yields ``None`` bounds and ``n_scenarios=0``; otherwise
    returns a percentile bootstrap CI on the mean.
    """
    if len(scenario_values) == 0:
        return {"mean_ci_lower": None, "mean_ci_upper": None, "mean_ci_n_scenarios": 0}
    lower, upper = bootstrap_ci(scenario_values, seed=seed)
    return {
        "mean_ci_lower": round(lower, decimals),
        "mean_ci_upper": round(upper, decimals),
        "mean_ci_n_scenarios": len(scenario_values),
    }
