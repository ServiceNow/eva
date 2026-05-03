#!/usr/bin/env python3
"""Faithfulness vs task_completion decoupling analysis.

For every clean trial across all systems, computes the joint distribution of
task_completion (binary) and faithfulness (quantized to {0, 0.5, 1.0} by the
judge) and reports two conditional probabilities:

  - P(faith < t | tc = 1): how often agents deviate even when the task succeeds
  - P(tc = 0    | faith < t): how often faithfulness failures co-occur with task failure

Two thresholds are reported because faithfulness is quantized:
  - 0.5 (substantial failure: only the 0.0 grade)
  - 1.0 (any deviation: 0.0 and 0.5 grades)

Outputs:
  output_processed/eva-bench-stats/decoupling/faithfulness_vs_task_completion.csv
  output_processed/eva-bench-stats/decoupling/faithfulness_vs_task_completion.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRIAL_SCORES_DIR = PROJECT_ROOT / "output" / "eva-bench-stats"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_processed" / "eva-bench-stats" / "decoupling"

THRESHOLDS = [0.5, 1.0]
THRESHOLD_LABEL = {
    0.5: "substantial failure (faith < 0.5)",
    1.0: "any deviation (faith < 1.0)",
}


def latest_trial_scores(data_dir: Path) -> Path:
    subs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("eva_clean_data_"))
    if not subs:
        raise FileNotFoundError(f"No eva_clean_data_* subdirectories in {data_dir}")
    return subs[-1] / "trial_scores.csv"


def _load_pairs(trial_scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trial_scores_path)
    df = df[df["perturbation_category"] == "clean"]
    df = df[df["metric"].isin(["task_completion", "faithfulness"])]
    keys = ["run_id", "system_alias", "system_type", "domain", "scenario_id", "trial"]
    wide = df.pivot_table(index=keys, columns="metric", values="value", aggfunc="first").reset_index()
    return wide.dropna(subset=["task_completion", "faithfulness"])


def _conditionals(wide: pd.DataFrame, threshold: float) -> dict:
    tc1 = wide[wide["task_completion"] == 1]
    fail = wide[wide["faithfulness"] < threshold]
    n = len(wide)
    n_tc1 = len(tc1)
    n_fail = len(fail)
    p_fail_given_tc1 = float((tc1["faithfulness"] < threshold).mean()) if n_tc1 else float("nan")
    p_tc0_given_fail = float((fail["task_completion"] == 0).mean()) if n_fail else float("nan")
    # Joint counts
    a = int(((wide["task_completion"] == 1) & (wide["faithfulness"] >= threshold)).sum())
    b = int(((wide["task_completion"] == 1) & (wide["faithfulness"] <  threshold)).sum())
    c = int(((wide["task_completion"] == 0) & (wide["faithfulness"] >= threshold)).sum())
    d = int(((wide["task_completion"] == 0) & (wide["faithfulness"] <  threshold)).sum())
    # Independence-baseline expected count for the (tc=1 & faith<t) cell
    expected_b = (n_tc1 * n_fail) / n if n else float("nan")
    return {
        "threshold": threshold,
        "label": THRESHOLD_LABEL[threshold],
        "n_total": n,
        "n_tc1": n_tc1,
        "n_faith_fail": n_fail,
        "n_tc1_and_faith_ok": a,
        "n_tc1_and_faith_fail": b,
        "n_tc0_and_faith_ok": c,
        "n_tc0_and_faith_fail": d,
        "p_faith_fail_given_tc1": p_fail_given_tc1,
        "p_tc0_given_faith_fail": p_tc0_given_fail,
        "expected_b_if_independent": expected_b,
    }


def _plot(rows: list[dict], wide: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel 1: faithfulness distribution within tc=0 vs tc=1
    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    for tc, color, label in [(1, "#2a9d8f", "task_completion = 1"),
                             (0, "#e76f51", "task_completion = 0")]:
        sub = wide[wide["task_completion"] == tc]["faithfulness"].to_numpy()
        ax.hist(sub, bins=bins, density=True, alpha=0.55, color=color, label=label,
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("faithfulness")
    ax.set_ylabel("density")
    ax.set_title("Faithfulness distribution by task outcome")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)

    # Panel 2: bar chart of the two key conditional probabilities at each threshold
    ax = axes[1]
    labels = [f"thresh = {r['threshold']}" for r in rows]
    p1 = [r["p_faith_fail_given_tc1"] for r in rows]
    p2 = [r["p_tc0_given_faith_fail"] for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    bars1 = ax.bar(x - width/2, p1, width, label="P(faith < t | tc = 1)", color="#3a86ff")
    bars2 = ax.bar(x + width/2, p2, width, label="P(tc = 0 | faith < t)", color="#7b2cbf")
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("conditional probability")
    ax.set_title("Decoupling between faithfulness and task completion")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Faithfulness vs task completion (all 12 systems, clean trials)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(trial_scores_path: Path, output_dir: Path) -> None:
    print(f"Loading {trial_scores_path}")
    wide = _load_pairs(trial_scores_path)
    print(f"  {len(wide):,} clean trials with both task_completion and faithfulness")
    print(f"  faithfulness value distribution:")
    print(wide["faithfulness"].value_counts(normalize=True).sort_index().round(4).to_string())

    rows = [_conditionals(wide, t) for t in THRESHOLDS]
    summary = pd.DataFrame(rows)
    print("\nConditional probabilities:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "faithfulness_vs_task_completion.csv"
    summary.to_csv(csv_path, index=False)
    pdf_path = output_dir / "faithfulness_vs_task_completion.pdf"
    _plot(rows, wide, pdf_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {pdf_path}")

    # Paper-ready sentences for each threshold
    print("\nPaper-ready (substantial failure, faith < 0.5):")
    r5 = next(r for r in rows if r["threshold"] == 0.5)
    print(
        f"  {r5['p_faith_fail_given_tc1']*100:.1f}% of trials with task_completion=1 "
        f"have faithfulness < 0.5; {r5['p_tc0_given_faith_fail']*100:.1f}% of "
        f"substantial faithfulness failures co-occur with task_completion=0 "
        f"(n_total = {r5['n_total']:,})."
    )
    print("\nPaper-ready (any deviation, faith < 1.0):")
    r1 = next(r for r in rows if r["threshold"] == 1.0)
    print(
        f"  {r1['p_faith_fail_given_tc1']*100:.1f}% of trials with task_completion=1 "
        f"exhibit at least one faithfulness deviation (faith < 1.0); "
        f"{r1['p_tc0_given_faith_fail']*100:.1f}% of faithfulness deviations "
        f"co-occur with task_completion=0 (n_total = {r1['n_total']:,})."
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
