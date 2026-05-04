# Config: local/eva-bench-stats/frontier_config.yaml
#
# output_dir: output_processed/eva-bench-stats/frontier
# random_seed: 42
# n_bootstrap: 1000
# quantiles: [0.75, 0.90]
# alpha: 0.05

"""Frontier analysis: quantile regression (Phase 1) and SFA stub (Phase 2).

Pure computation: reads DataFrames, returns DataFrames. No file I/O, no plotting.

Phase 1 pipeline:
  fit_quantile_regression  -- fits one QR model at a given quantile and direction,
                              with bootstrapped slope CI from stats_utils
  run_quantile_regression  -- fits all 4 models (2 quantiles x 2 directions)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.regression.quantile_regression import QuantReg

from stats_utils import bootstrap_slope_ci

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "frontier_config.yaml"


def fit_quantile_regression(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float,
    x_name: str,
    y_name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Fit quantile regression modeling the conditional quantile of y given x.

    A negative slope means high x is associated with a lower ceiling on y --
    a frontier trade-off. Bootstrapped CIs are used because analytical SEs
    for QR assume n >> 11.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])

    converged = True
    slope = intercept = p_value = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = QuantReg(y, X).fit(q=quantile, max_iter=2000)
        intercept = float(result.params[0])
        slope = float(result.params[1])
        p_value = float(result.pvalues[1])
    except Exception as exc:
        print(f"WARNING: QR failed for {y_name} ~ {x_name} at q={quantile}: {exc}")
        converged = False

    ci_lower, ci_upper = bootstrap_slope_ci(x, y, quantile=quantile, n_boot=n_boot, seed=seed)

    return {
        "quantile": quantile,
        "x_name": x_name,
        "y_name": y_name,
        "slope": slope,
        "intercept": intercept,
        "slope_ci_lower": ci_lower,
        "slope_ci_upper": ci_upper,
        "p_value": p_value,
        "converged": converged,
        "n": n,
    }


def run_quantile_regression(scores_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run all 4 QR models (2 quantiles x 2 directions) and return as DataFrame."""
    n = len(scores_df)
    print(
        f"WARNING: n={n} models is small. QR results should be treated as "
        "exploratory, not confirmatory."
    )

    quantiles: list[float] = config["quantiles"]
    n_boot: int = config["n_bootstrap"]
    seed: int = config["random_seed"]

    eva_a = scores_df["eva_a"].to_numpy()
    eva_x = scores_df["eva_x"].to_numpy()

    directions = [
        (eva_a, eva_x, "EVA-A", "EVA-X"),
        (eva_x, eva_a, "EVA-X", "EVA-A"),
    ]

    rows: list[dict] = []
    for q in quantiles:
        for x_vals, y_vals, x_name, y_name in directions:
            print(f"  Fitting QR: {y_name} ~ {x_name}, q={q} ...")
            row = fit_quantile_regression(
                x_vals, y_vals,
                quantile=q, x_name=x_name, y_name=y_name,
                n_boot=n_boot, seed=seed,
            )
            rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            "quantile", "x_name", "y_name", "slope", "intercept",
            "slope_ci_lower", "slope_ci_upper", "p_value", "converged", "n",
        ],
    )


def compute_frontier_stats(scores_df: pd.DataFrame, config: dict) -> dict:
    """Phase 2 stub: stochastic frontier analysis (SFA). Not yet implemented."""
    raise NotImplementedError("SFA not yet implemented -- see Phase 2 spec.")


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"]

    scores_path = output_dir / "model_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"model_scores.csv not found at {scores_path}. Run run_data.py first."
        )

    print(f"Loading model scores from {scores_path} ...")
    scores_df = pd.read_csv(scores_path)
    print(f"  {len(scores_df)} models loaded")

    print("Running quantile regression ...")
    qr_results = run_quantile_regression(scores_df, config)

    qr_path = output_dir / "qr_results.csv"
    qr_results.to_csv(qr_path, index=False)
    print(f"Wrote {len(qr_results)} rows -> {qr_path}")


if __name__ == "__main__":
    main()
