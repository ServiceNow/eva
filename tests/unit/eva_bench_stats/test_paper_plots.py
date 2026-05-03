from pathlib import Path
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_plots import asymmetric_err


def test_asymmetric_err_returns_nonneg_pair() -> None:
    lo_arr, hi_arr = asymmetric_err([0.5, 0.2], [0.4, 0.15], [0.55, 0.30])
    np.testing.assert_allclose(lo_arr, [0.10, 0.05])
    np.testing.assert_allclose(hi_arr, [0.05, 0.10])


def test_asymmetric_err_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        asymmetric_err([0.5], [0.6], [0.55])
