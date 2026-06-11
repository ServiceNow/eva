"""Bootstrap primitives for sample-mean confidence intervals."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

N_BOOT = 2000
ALPHA = 0.05


def run_seed(run_id: str) -> int:
    """Seed from ``run_id`` using SHA-256.

    Process-stable unlike Python's ``hash()``, so CI bounds are consistent across
    ``eva metrics`` invocations on the same run.
    """
    h = hashlib.sha256(run_id.encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def bootstrap_resample(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Return ``n_boot`` resampled means of ``values``."""
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
) -> tuple[float | None, float | None]:
    """95% bootstrap CI on the mean (default alpha=0.05)."""
    if len(values) == 0:
        return None, None
    boot = bootstrap_resample(values, n_boot=n_boot, seed=seed)
    lower = float(np.percentile(boot, 100 * alpha / 2))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lower, upper


def named_ci_fields(
    samples: dict[str, Sequence[float]],
    *,
    seed: int,
    decimals: int = 4,
) -> dict[str, float | None]:
    """Percentile bootstrap CI on the mean of each named metric in ``samples``."""
    out: dict[str, float | None] = {}
    for name, sample in samples.items():
        if not sample:
            out[f"{name}_ci_lower"] = None
            out[f"{name}_ci_upper"] = None
            continue
        lower, upper = bootstrap_ci(sample, seed=seed)
        out[f"{name}_ci_lower"] = round(lower, decimals) if lower is not None else None
        out[f"{name}_ci_upper"] = round(upper, decimals) if upper is not None else None
    return out


def mean_ci_fields(
    scenario_values: Sequence[float],
    *,
    seed: int,
    decimals: int = 4,
) -> dict[str, Any]:
    """Percentile bootstrap CI on the mean of ``scenario_values``, plus scenario count."""
    if len(scenario_values) == 0:
        return {"mean_ci_lower": None, "mean_ci_upper": None, "mean_ci_n_scenarios": 0}
    lower, upper = bootstrap_ci(scenario_values, seed=seed)
    return {
        "mean_ci_lower": round(lower, decimals),
        "mean_ci_upper": round(upper, decimals),
        "mean_ci_n_scenarios": len(scenario_values),
    }
