import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_tables import format_cell, shade_index, lookup_pooled


def test_format_cell_symmetric() -> None:
    assert format_cell(0.428, 0.393, 0.463) == "0.428 $\\pm$0.035"


def test_format_cell_asymmetric_uses_max_halfwidth() -> None:
    # point - lo = 0.080, hi - point = 0.020 → max = 0.080
    assert format_cell(0.500, 0.420, 0.520) == "0.500 $\\pm$0.080"


def test_format_cell_missing() -> None:
    assert format_cell(None, None, None) == "--"
    assert format_cell(float("nan"), float("nan"), float("nan")) == "--"


def test_shade_index_clamps_and_buckets() -> None:
    # min=0.1, max=0.5 → 7 even buckets across range
    assert shade_index(0.1, lo=0.1, hi=0.5, n_steps=7) == 0
    assert shade_index(0.5, lo=0.1, hi=0.5, n_steps=7) == 6
    assert shade_index(0.05, lo=0.1, hi=0.5, n_steps=7) == 0  # clamp low
    assert shade_index(0.6, lo=0.1, hi=0.5, n_steps=7) == 6   # clamp high
    # degenerate range
    assert shade_index(0.3, lo=0.3, hi=0.3, n_steps=7) == 0


def _pooled_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"model_label": "M1", "metric": "EVA-A_pass", "domain": "pooled",
         "point_estimate": 0.40, "ci_lower": 0.35, "ci_upper": 0.45},
        {"model_label": "M1", "metric": "task_completion", "domain": "pooled",
         "point_estimate": 0.60, "ci_lower": 0.55, "ci_upper": 0.65},
        # non-pooled rows must be ignored
        {"model_label": "M1", "metric": "EVA-A_pass", "domain": "itsm",
         "point_estimate": 0.99, "ci_lower": 0.98, "ci_upper": 1.0},
    ])


def test_lookup_pooled_hits() -> None:
    df = _pooled_df()
    point, lo, hi = lookup_pooled(df, "M1", "EVA-A_pass")
    assert (point, lo, hi) == (0.40, 0.35, 0.45)


def test_lookup_pooled_misses_returns_nones() -> None:
    df = _pooled_df()
    assert lookup_pooled(df, "M1", "EVA-A_pass_at_k") == (None, None, None)
    assert lookup_pooled(df, "Ghost", "EVA-A_pass") == (None, None, None)
