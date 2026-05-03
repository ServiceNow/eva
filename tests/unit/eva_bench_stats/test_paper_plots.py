from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_plots import asymmetric_err, build_scatter_points, pareto_frontier_indices, write_scatter
from paper_config import ModelEntry, PaperConfig


def test_asymmetric_err_returns_nonneg_pair() -> None:
    lo_arr, hi_arr = asymmetric_err([0.5, 0.2], [0.4, 0.15], [0.55, 0.30])
    np.testing.assert_allclose(lo_arr, [0.10, 0.05])
    np.testing.assert_allclose(hi_arr, [0.05, 0.10])


def test_asymmetric_err_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        asymmetric_err([0.5], [0.6], [0.55])


def _cfg_two_models() -> PaperConfig:
    return PaperConfig(
        output_dir="ignored",
        accuracy_aggregate={}, accuracy_submetrics={},
        experience_aggregate={}, experience_submetrics={},
        scatter={"pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"}},
        models={
            "Sys A": ModelEntry(label="Sys A", alias="a", arch="cascade"),
            "Sys B": ModelEntry(label="Sys B", alias="b", arch="s2s"),
        },
    )


def _pooled_full() -> pd.DataFrame:
    rows = []
    for m, ax, ay in [("Sys A", 0.40, 0.50), ("Sys B", 0.62, 0.30)]:
        rows.append({"model_label": m, "metric": "EVA-A_pass", "domain": "pooled",
                     "point_estimate": ax, "ci_lower": ax - 0.04, "ci_upper": ax + 0.05})
        rows.append({"model_label": m, "metric": "EVA-X_pass", "domain": "pooled",
                     "point_estimate": ay, "ci_lower": ay - 0.03, "ci_upper": ay + 0.06})
    return pd.DataFrame(rows)


def test_build_scatter_points_full() -> None:
    pts = build_scatter_points(_pooled_full(), _cfg_two_models(), "EVA-A_pass", "EVA-X_pass")
    labels = {p.label for p in pts}
    assert labels == {"Sys A", "Sys B"}
    a = next(p for p in pts if p.label == "Sys A")
    assert (a.x, a.x_lo, a.x_hi) == pytest.approx((0.40, 0.36, 0.45))
    assert (a.y, a.y_lo, a.y_hi) == pytest.approx((0.50, 0.47, 0.56))


def test_build_scatter_points_drops_missing() -> None:
    df = _pooled_full().drop(_pooled_full().index[0])  # drop Sys A x-row
    pts = build_scatter_points(df, _cfg_two_models(), "EVA-A_pass", "EVA-X_pass")
    assert {p.label for p in pts} == {"Sys B"}


def test_pareto_frontier_indices_simple() -> None:
    # Points: (x, y). Higher-better on both. Frontier = {(0.6, 0.5), (0.4, 0.7), (0.5, 0.55)}.
    pts = [(0.6, 0.5), (0.4, 0.7), (0.3, 0.4), (0.5, 0.55)]
    idx = pareto_frontier_indices(pts)
    assert sorted(idx) == [0, 1, 3]


def test_pareto_frontier_indices_empty() -> None:
    assert pareto_frontier_indices([]) == []


def test_write_scatter_creates_pdf(tmp_path: Path) -> None:
    out = tmp_path / "scatter.pdf"
    n_drawn = write_scatter(
        _pooled_full(), _cfg_two_models(),
        x_metric="EVA-A_pass", y_metric="EVA-X_pass",
        x_label="Accuracy (EVA-A pass@1)",
        y_label="Experience (EVA-X pass@1)",
        title="Accuracy vs Experience pass@1",
        out_path=out,
    )
    assert n_drawn == 2
    assert out.exists() and out.stat().st_size > 0


def test_write_scatter_returns_zero_when_no_points(tmp_path: Path) -> None:
    out = tmp_path / "skip.pdf"
    df = pd.DataFrame(columns=["model_label", "metric", "domain",
                               "point_estimate", "ci_lower", "ci_upper"])
    n_drawn = write_scatter(
        df, _cfg_two_models(),
        x_metric="EVA-A_pass_at_k", y_metric="EVA-X_pass_at_k",
        x_label="x", y_label="y", title="t", out_path=out,
    )
    assert n_drawn == 0
    assert not out.exists()
