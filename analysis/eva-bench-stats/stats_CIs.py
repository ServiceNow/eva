"""Bootstrapped confidence intervals on scenario-level means for clean runs.

Pipeline:
  1. stability_check       — bootstrap @ 1000 vs 2000 on a representative model;
                             choose n_boot for full run.
  2. compute_domain_ci     — per (model, metric, domain): point + percentile CI
                             from scenario-level bootstrap.
  3. compute_pooled_ci     — equal-weighted pooled CI: mean of 3 domain points,
                             percentiles of elementwise mean of 3 domain bootstraps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from stats_utils import bootstrap_resample  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "CIs_config.yaml"


def _cell_seed(base_seed: int, *parts: str) -> int:
    return (base_seed + hash(":".join(parts))) % (2**31)


def compute_domain_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, np.ndarray]:
    """(point_estimate, ci_lower, ci_upper, boot_dist) for one model × metric × domain."""
    values = np.asarray(values, dtype=float)
    point = float(values.mean()) if len(values) else float("nan")
    boot = bootstrap_resample(values, n_boot=n_boot, seed=seed)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi, boot


def compute_pooled_ci(
    domain_points: list[float],
    domain_dists: list[np.ndarray],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Equal-weighted pooled CI.

    Pooled point: mean of the 3 domain point estimates.
    Pooled CI:   percentile of the elementwise mean of the 3 bootstrap distributions.
    """
    point = float(np.mean(domain_points))
    stacked = np.vstack(domain_dists)             # shape (n_domains, n_boot)
    pooled_dist = stacked.mean(axis=0)            # shape (n_boot,)
    lo = float(np.percentile(pooled_dist, 100 * alpha / 2))
    hi = float(np.percentile(pooled_dist, 100 * (1 - alpha / 2)))
    return point, lo, hi


def stability_check(
    scenario_means_df: pd.DataFrame,
    model_label: str,
    metrics: list[str],
    expected_domains: list[str],
    base_seed: int,
    alpha: float,
    threshold: float,
    n_low: int = 1000,
    n_high: int = 2000,
) -> tuple[int, pd.DataFrame]:
    """Run bootstrap @ n_low and n_high for one model, all metrics × all domains.

    Returns (chosen_n_boot, log_df). log_df has one row per metric with
    max |Δ CI bound| across domains and a within_tolerance flag.
    """
    sub = scenario_means_df[scenario_means_df["model_label"] == model_label]
    rows: list[dict] = []
    overall_max = 0.0

    for metric in metrics:
        diffs: list[float] = []
        for domain in expected_domains:
            cell = sub[(sub["metric"] == metric) & (sub["domain"] == domain)]
            if cell.empty:
                continue
            x = cell["scenario_mean"].to_numpy()
            seed = _cell_seed(base_seed, "stability", model_label, metric, domain)
            _, lo_a, hi_a, _ = compute_domain_ci(x, n_boot=n_low, seed=seed, alpha=alpha)
            _, lo_b, hi_b, _ = compute_domain_ci(x, n_boot=n_high, seed=seed, alpha=alpha)
            diffs.append(max(abs(lo_a - lo_b), abs(hi_a - hi_b)))
        max_diff = max(diffs) if diffs else 0.0
        overall_max = max(overall_max, max_diff)
        rows.append({
            "metric": metric,
            "n_boot_ref": n_low,
            "n_boot_test": n_high,
            "max_abs_ci_diff": max_diff,
            "within_tolerance": max_diff <= threshold,
        })

    chosen = n_low if overall_max <= threshold else n_high
    log_df = pd.DataFrame(rows)
    log_df.attrs["overall_max"] = overall_max
    log_df.attrs["chosen_n_boot"] = chosen
    return chosen, log_df


def run_analysis(
    scenario_means_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run stability check + per-domain CIs + pooled CIs.

    Returns (per_domain_df, pooled_df, stability_log_df).
    """
    metrics: list[str] = config["metrics"]
    expected_domains: list[str] = config["expected_domains"]
    base_seed: int = config["random_seed"]
    alpha: float = config["alpha"]
    n_boot_default: int = config["n_bootstrap"]
    threshold: float = config["stability_threshold"]

    # Choose stability check model
    models = scenario_means_df["model_label"].drop_duplicates().tolist()
    if not models:
        empty = pd.DataFrame(columns=["model_label", "metric", "domain", "n", "point_estimate", "ci_lower", "ci_upper"])
        return empty, empty.copy(), pd.DataFrame()
    stability_model = config.get("stability_check_model") or models[0]
    if stability_model not in models:
        print(f"  [stability] configured model '{stability_model}' not in data; falling back to '{models[0]}'")
        stability_model = models[0]

    print(f"  [stability] running 1000 vs 2000 on '{stability_model}' ...")
    chosen_n_boot, stability_log = stability_check(
        scenario_means_df, stability_model, metrics, expected_domains,
        base_seed=base_seed, alpha=alpha, threshold=threshold,
    )
    if chosen_n_boot != n_boot_default:
        print(f"  [stability] max Δ = {stability_log.attrs['overall_max']:.5f} > {threshold} → using n_boot = {chosen_n_boot}")
    else:
        print(f"  [stability] max Δ = {stability_log.attrs['overall_max']:.5f} ≤ {threshold} → using n_boot = {chosen_n_boot}")

    per_domain_rows: list[dict] = []
    pooled_rows: list[dict] = []

    for model_label in models:
        model_df = scenario_means_df[scenario_means_df["model_label"] == model_label]
        for metric in metrics:
            domain_dists: list[np.ndarray] = []
            domain_points: list[float] = []
            domains_present: list[str] = []

            for domain in expected_domains:
                cell = model_df[(model_df["metric"] == metric) & (model_df["domain"] == domain)]
                if cell.empty:
                    continue
                x = cell["scenario_mean"].to_numpy()
                seed = _cell_seed(base_seed, "main", model_label, metric, domain)
                point, lo, hi, boot = compute_domain_ci(x, n_boot=chosen_n_boot, seed=seed, alpha=alpha)
                per_domain_rows.append({
                    "model_label": model_label,
                    "metric": metric,
                    "domain": domain,
                    "n": len(x),
                    "point_estimate": point,
                    "ci_lower": lo,
                    "ci_upper": hi,
                })
                domain_dists.append(boot)
                domain_points.append(point)
                domains_present.append(domain)

            if len(domain_dists) == len(expected_domains):
                p_point, p_lo, p_hi = compute_pooled_ci(domain_points, domain_dists, alpha=alpha)
                pooled_rows.append({
                    "model_label": model_label,
                    "metric": metric,
                    "domain": "pooled",
                    "n": "pooled",
                    "point_estimate": p_point,
                    "ci_lower": p_lo,
                    "ci_upper": p_hi,
                })
            else:
                missing = [d for d in expected_domains if d not in domains_present]
                print(f"  [skip pooled] {model_label} × {metric}: missing {missing}")

    per_domain_df = pd.DataFrame(per_domain_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    return per_domain_df, pooled_df, stability_log


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"]
    data_dir = output_dir / "data"
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    means_path = data_dir / "scenario_means.csv"
    if not means_path.exists():
        raise FileNotFoundError(f"scenario_means.csv not found at {means_path}. Run data_CIs.py first.")

    print(f"Loading scenario means from {means_path} ...")
    means_df = pd.read_csv(means_path)
    print(f"  {len(means_df):,} rows loaded")

    per_domain_df, pooled_df, stability_log = run_analysis(means_df, config)

    per_domain_df.to_csv(stats_dir / "results_per_domain.csv", index=False)
    pooled_df.to_csv(stats_dir / "results_pooled.csv", index=False)
    stability_log.to_csv(stats_dir / "stability_log.csv", index=False)

    print(f"\nWrote {len(per_domain_df):,} per-domain rows → {stats_dir / 'results_per_domain.csv'}")
    print(f"Wrote {len(pooled_df):,} pooled rows      → {stats_dir / 'results_pooled.csv'}")
    print(f"Wrote stability log                       → {stats_dir / 'stability_log.csv'}")


if __name__ == "__main__":
    main()
