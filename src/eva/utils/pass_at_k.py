"""pass@k and pass^k computation for multi-trial evaluation.

Provides mathematical functions and utilities for computing pass@k and pass^k
metrics across multiple trials of the same evaluation record.

- pass@k: Probability that at least 1 of k randomly drawn samples passes.
- pass^k: Probability that all k randomly drawn samples pass.
"""

import re
from math import comb

from eva.models.results import MetricScore, PassAtKResult

# Regex pattern to extract trial index from nested path like "record_id/trial_0"
NESTED_TRIAL_PATTERN = re.compile(r"/trial_(\d+)$")

# Regex pattern to extract trial index from flat directory names like "record_id_trial_0" (backward compat)
TRIAL_SUFFIX_PATTERN = re.compile(r"_trial_(\d+)$")

# Regex pattern to strip attempt suffix from directory names like "record_id_attempt_1"
ATTEMPT_SUFFIX_PATTERN = re.compile(r"_attempt_(\d+)$")


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k: probability that at least 1 of k draws passes.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of trials.
        c: Number of passing trials.
        k: Number of draws.

    Returns:
        pass@k probability in [0.0, 1.0].

    Raises:
        ValueError: If k > n or any value is negative.
    """
    if k < 0 or n < 0 or c < 0:
        raise ValueError(f"All values must be non-negative: n={n}, c={c}, k={k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c > n:
        raise ValueError(f"c ({c}) cannot exceed n ({n})")
    if k == 0:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_power_k(n: int, c: int, k: int) -> float:
    """Compute pass^k: probability that all k draws pass.

    Formula: pass^k = C(c, k) / C(n, k)

    Args:
        n: Total number of trials.
        c: Number of passing trials.
        k: Number of draws.

    Returns:
        pass^k probability in [0.0, 1.0].

    Raises:
        ValueError: If k > n or any value is negative.
    """
    if k < 0 or n < 0 or c < 0:
        raise ValueError(f"All values must be non-negative: n={n}, c={c}, k={k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c > n:
        raise ValueError(f"c ({c}) cannot exceed n ({n})")
    if k == 0:
        return 1.0
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def parse_trial_record_id(dir_name: str) -> tuple[str, int | None]:
    """Parse a directory name to extract base record ID and trial index.

    Handles the naming conventions:
    - "{record_id}/trial_{N}" (nested, preferred)
    - "{record_id}/trial_{N}_attempt_{M}" (nested with attempt suffix)
    - "{record_id}_trial_{N}" (flat, backward compat)
    - "{record_id}_trial_{N}_attempt_{M}" (flat with attempt suffix)
    - "{record_id}_attempt_{M}" (strips attempt suffix, no trial)

    Args:
        dir_name: Directory name, e.g. "1.2.1/trial_0", "1.2.1_trial_0", or "1.2.1".

    Returns:
        Tuple of (base_record_id, trial_index). trial_index is None
        if the directory name does not match the trial pattern.

    Examples:
        >>> parse_trial_record_id("1.2.1/trial_0")
        ("1.2.1", 0)
        >>> parse_trial_record_id("1.2.1_trial_0")
        ("1.2.1", 0)
        >>> parse_trial_record_id("1.2.1")
        ("1.2.1", None)
        >>> parse_trial_record_id("my_record_trial_3")
        ("my_record", 3)
        >>> parse_trial_record_id("1.2.1_trial_0_attempt_1")
        ("1.2.1", 0)
        >>> parse_trial_record_id("1.2.1/trial_0_attempt_1")
        ("1.2.1", 0)
        >>> parse_trial_record_id("1.2.1_attempt_2")
        ("1.2.1", None)
    """
    # Strip attempt suffix first if present
    name = dir_name
    attempt_match = ATTEMPT_SUFFIX_PATTERN.search(name)
    if attempt_match:
        name = name[: attempt_match.start()]

    # Try nested path pattern first: "record_id/trial_N"
    nested_match = NESTED_TRIAL_PATTERN.search(name)
    if nested_match:
        base_id = name[: nested_match.start()]
        trial_idx = int(nested_match.group(1))
        return base_id, trial_idx

    # Fall back to flat pattern: "record_id_trial_N"
    match = TRIAL_SUFFIX_PATTERN.search(name)
    if match:
        base_id = name[: match.start()]
        trial_idx = int(match.group(1))
        return base_id, trial_idx
    return name, None


def compute_pass_at_k_for_scores(
    metric_name: str,
    per_trial_scores: list[MetricScore],
    threshold: float,
    k: int,
) -> PassAtKResult | None:
    """Compute pass@k and pass^k from a list of per-trial MetricScores.

    Errored trials are excluded entirely. If fewer than k valid trials remain,
    returns None (the record should not participate in pass@k for this metric).

    Args:
        metric_name: Name of the metric being evaluated.
        per_trial_scores: MetricScore objects from each trial.
        threshold: Score threshold to determine pass/fail (uses normalized_score).
        k: Number of draws for pass@k computation.

    Returns:
        PassAtKResult with computed pass@k and pass^k values, or None if
        there are fewer than k valid (non-errored) trials.
    """
    valid_scores: list[float] = []
    valid_passed: list[bool] = []

    for ms in per_trial_scores:
        if ms.error is not None:
            continue
        # Skipped trials contribute no pass/fail signal to pass@k — exclude them.
        if ms.skipped:
            continue
        val = ms.normalized_score if ms.normalized_score is not None else ms.score
        valid_scores.append(val)
        valid_passed.append(val >= threshold)

    n = len(valid_scores)
    if n < k:
        return None

    c = sum(valid_passed)

    return PassAtKResult(
        metric_name=metric_name,
        n=n,
        k=k,
        c=c,
        pass_at_k=compute_pass_at_k(n, c, k),
        pass_power_k=compute_pass_power_k(n, c, k),
        threshold=threshold,
        per_trial_scores=valid_scores,
        per_trial_passed=valid_passed,
    )
