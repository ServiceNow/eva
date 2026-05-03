#!/usr/bin/env python3
"""Cascade-only correlation: transcription accuracy on key entities vs task completion.

Computes Pearson r and p across all (system, scenario, trial) triples for the
clean cascade runs, plus a per-system breakdown. Writes a CSV summary and prints
a paper-ready sentence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRIAL_SCORES_DIR = PROJECT_ROOT / "output" / "eva-bench-stats"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_processed" / "eva-bench-stats" / "correlations"

X_METRIC = "transcription_accuracy_key_entities"
Y_METRIC = "task_completion"


def latest_trial_scores(data_dir: Path) -> Path:
    subs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("eva_clean_data_"))
    if not subs:
        raise FileNotFoundError(f"No eva_clean_data_* subdirectories in {data_dir}")
    return subs[-1] / "trial_scores.csv"


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    res = stats.pearsonr(x, y)
    return float(res.statistic), float(res.pvalue), n


def run(trial_scores_path: Path, output_dir: Path) -> None:
    print(f"Loading {trial_scores_path}")
    df = pd.read_csv(trial_scores_path)
    df = df[(df["perturbation_category"] == "clean") & (df["system_type"] == "cascade")]
    df = df[df["metric"].isin([X_METRIC, Y_METRIC])]

    keys = ["run_id", "system_alias", "domain", "scenario_id", "trial"]
    wide = df.pivot_table(index=keys, columns="metric", values="value", aggfunc="first").reset_index()
    wide = wide.dropna(subset=[X_METRIC, Y_METRIC])
    print(f"  {len(wide):,} cascade trials with both {X_METRIC} and {Y_METRIC}")

    rows: list[dict] = []
    for alias, g in wide.groupby("system_alias", sort=True):
        r, p, n = pearson(g[X_METRIC].to_numpy(), g[Y_METRIC].to_numpy())
        rows.append({"system_alias": alias, "n": n, "r": r, "p": p})

    r_all, p_all, n_all = pearson(wide[X_METRIC].to_numpy(), wide[Y_METRIC].to_numpy())
    rows.append({"system_alias": "combined", "n": n_all, "r": r_all, "p": p_all})

    summary = pd.DataFrame(rows)
    print("\nPer-system Pearson:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "transcription_vs_task_completion.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    if np.isnan(r_all):
        print("\n[combined] insufficient data for correlation")
        return
    p_str = f"{p_all:.2e}" if p_all < 1e-3 else f"{p_all:.3f}"
    print("\nPaper-ready:")
    print(
        f"  Across cascade systems, transcription accuracy on key entities correlates "
        f"with task completion at the trial level (Pearson r = {r_all:.3f}, "
        f"p = {p_str}, n = {n_all:,})."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-scores", type=Path,
                    help="Path to trial_scores.csv. Defaults to latest eva_clean_data_*.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    path = args.trial_scores or latest_trial_scores(DEFAULT_TRIAL_SCORES_DIR)
    run(path, args.output_dir)


if __name__ == "__main__":
    main()
