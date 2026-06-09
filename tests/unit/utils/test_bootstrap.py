"""Unit tests for src/eva/utils/bootstrap.py."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import numpy as np

from eva.utils.bootstrap import (
    ALPHA,
    N_BOOT,
    bootstrap_ci,
    bootstrap_resample,
    run_seed,
)


class TestBootstrapResample:
    def test_shape_and_determinism(self):
        values = np.array([0.0, 0.5, 1.0, 0.25, 0.75])
        a = bootstrap_resample(values, n_boot=100, seed=42)
        b = bootstrap_resample(values, n_boot=100, seed=42)
        assert a.shape == (100,)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        values = np.array([0.0, 0.5, 1.0])
        a = bootstrap_resample(values, n_boot=100, seed=1)
        b = bootstrap_resample(values, n_boot=100, seed=2)
        assert not np.array_equal(a, b)

    def test_constant_input_constant_output(self):
        values = np.full(10, 0.7)
        boot = bootstrap_resample(values, n_boot=50, seed=0)
        np.testing.assert_allclose(boot, 0.7)

    def test_empty_input(self):
        boot = bootstrap_resample(np.array([]), n_boot=10, seed=0)
        assert boot.shape == (0,)


class TestBootstrapCI:
    def test_brackets_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=0.5, scale=0.1, size=100)
        lower, upper = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)
        assert lower < values.mean() < upper
        assert upper - lower < 0.1

    def test_narrower_alpha_widens(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=0.5, scale=0.1, size=100)
        lo_90, hi_90 = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.10)
        lo_95, hi_95 = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)
        assert (hi_95 - lo_95) > (hi_90 - lo_90)

    def test_empty_input_returns_nones(self):
        lower, upper = bootstrap_ci(np.array([]), n_boot=100, seed=0)
        assert lower is None
        assert upper is None

    def test_single_value(self):
        lower, upper = bootstrap_ci(np.array([0.42]), n_boot=100, seed=0)
        assert lower == upper == 0.42

    def test_n_boot_and_alpha_defaults_match_module_constants(self):
        # bootstrap_ci's optional n_boot/alpha defaults should match the module constants.
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        a = bootstrap_ci(values, seed=0)
        b = bootstrap_ci(values, n_boot=N_BOOT, seed=0, alpha=ALPHA)
        assert a == b


class TestRunSeed:
    def test_deterministic_same_input(self):
        assert run_seed("abc") == run_seed("abc")

    def test_different_inputs_differ(self):
        assert run_seed("abc") != run_seed("def")

    def test_returns_nonnegative_int(self):
        s = run_seed("any-run-id")
        assert isinstance(s, int)
        assert s >= 0
        assert s < 2**31

    def test_cross_process_stable(self):
        """run_seed must NOT use Python's salted hash(); spawn a subprocess and check equality."""
        in_process = run_seed("cross-process-check")
        script = textwrap.dedent(
            """
            from eva.utils.bootstrap import run_seed
            print(run_seed("cross-process-check"))
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        subprocess_value = int(result.stdout.strip())
        assert in_process == subprocess_value
