# analysis/eva-bench-stats/data_variance.py
# Config: local/eva-bench-stats/variance_config.yaml
#
# output_dir: output_processed/eva-bench-stats/variance
# metrics: [EVA-A_mean, EVA-X_mean, EVA-overall_mean, task_completion,
#           faithfulness, agent_speech_fidelity, conversation_progression, conciseness]
# runs:
#   "<display_label>":
#     run_id: <run_id>
#     type: cascade  # or s2s
#     stt/llm/tts or s2s/voice keys depending on type
"""Load and process iteration-archive data for variance analysis.

Pipeline:
  load_scores_iter / load_aggregate_scores_iter  → scores_df, agg_df
  compute_judge_variance        → judge_var.csv
  compute_trial_variance        → trial_var.csv
  compute_judge_variance_summary → judge_summary.csv
  compute_trial_variance_summary → trial_summary.csv
  compute_composite_stability   → composite_stability.csv
  threshold_crossings           → borderline_scenarios.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "variance_config.yaml"
ARCHIVE_ROOT = PROJECT_ROOT / "output" / "judge_variance_analysis"


def compute_judge_variance(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per (run_id, metric, record_id, trial): std dev, range, score_changed across iterations."""
    filtered = df[df["metric"].isin(metrics)]
    grouped = filtered.groupby(["run_id", "run_label", "metric", "record_id", "trial"])["normalized_score"]
    result = grouped.agg(
        std=lambda x: float(np.std(x, ddof=0)),
        range=lambda x: float(x.max() - x.min()),
    ).reset_index()
    nunique = grouped.nunique().reset_index(name="nunique")
    result["score_changed"] = pd.array([bool(v > 1) for v in nunique["nunique"]], dtype=object)
    return result


def compute_trial_variance(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per (run_id, metric, record_id): std dev and range across trials, judge noise removed.

    Judge noise is removed by averaging over iterations before computing cross-trial std dev.
    """
    filtered = df[df["metric"].isin(metrics)]
    mean_by_trial = (
        filtered.groupby(["run_id", "run_label", "metric", "record_id", "trial"])["normalized_score"]
        .mean()
        .reset_index()
        .rename(columns={"normalized_score": "mean_score"})
    )
    return (
        mean_by_trial.groupby(["run_id", "run_label", "metric", "record_id"])["mean_score"]
        .agg(
            std=lambda x: float(np.std(x, ddof=0)),
            range=lambda x: float(x.max() - x.min()),
        )
        .reset_index()
    )


def compute_judge_variance_summary(judge_var: pd.DataFrame) -> pd.DataFrame:
    """Aggregate judge variance stats per (run_id, metric)."""
    return (
        judge_var.groupby(["run_id", "run_label", "metric"])
        .agg(
            mean_std=("std", "mean"),
            std_of_std=("std", "std"),
            std_min=("std", "min"),
            std_max=("std", "max"),
            mean_range=("range", "mean"),
            pct_changed=("score_changed", "mean"),
            n=("std", "count"),
        )
        .reset_index()
    )


def compute_trial_variance_summary(trial_var: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial variance stats per (run_id, metric)."""
    return (
        trial_var.groupby(["run_id", "run_label", "metric"])
        .agg(
            mean_std=("std", "mean"),
            std_of_std=("std", "std"),
            std_min=("std", "min"),
            std_max=("std", "max"),
            mean_range=("range", "mean"),
            n=("std", "count"),
        )
        .reset_index()
    )


def compute_composite_stability(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Per (run_id, iteration): pass@1, pass@k, pass^k, and mean composites."""
    K = 3
    PASS_THRESHOLD = 1.0

    pass_composites = [c for c in ["EVA-overall_pass", "EVA-A_pass", "EVA-X_pass"] if c in agg_df.columns]
    mean_composites = [c for c in ["EVA-A_mean", "EVA-X_mean", "EVA-overall_mean"] if c in agg_df.columns]

    results = []
    for (run_id, run_label, iteration), grp in agg_df.groupby(["run_id", "run_label", "iteration"]):
        row: dict = {"run_id": run_id, "run_label": run_label, "iteration": iteration}

        for composite in pass_composites:
            scenario_stats = []
            for _record_id, scenario_grp in grp.groupby("record_id"):
                vals = scenario_grp[composite].dropna().values
                n = len(vals)
                if n == 0:
                    continue
                c = int(np.sum(vals >= PASS_THRESHOLD))
                scenario_stats.append(
                    {
                        "pass_at_1": c / n,
                        "pass_at_k": 1.0 if c >= 1 else 0.0,
                        "pass_power_k": (c / n) ** K,
                    }
                )
            if scenario_stats:
                s = pd.DataFrame(scenario_stats)
                col_base = composite.removesuffix("_pass")
                row[f"{col_base}_pass_at_1"] = float(s["pass_at_1"].mean())
                row[f"{col_base}_pass_at_k"] = float(s["pass_at_k"].mean())
                row[f"{col_base}_pass_power_k"] = float(s["pass_power_k"].mean())

        for composite in mean_composites:
            if composite in grp.columns:
                row[composite] = float(grp[composite].mean())

        results.append(row)

    return pd.DataFrame(results)


def threshold_crossings(scores_df: pd.DataFrame, pass_thresholds: dict[str, float]) -> pd.DataFrame:
    """Rows where judge stochasticity flipped a score across an EVA pass/fail threshold.

    For each (run_id, metric, record_id, trial): if min score across iterations is
    below the threshold AND max is at or above it, that trial had a pass/fail flip.
    """
    rows = []
    for metric, threshold in pass_thresholds.items():
        sub = scores_df[scores_df["metric"] == metric]
        for (run_id, run_label, record_id, trial), grp in sub.groupby(["run_id", "run_label", "record_id", "trial"]):
            s = grp["normalized_score"].values
            if len(s) < 2:
                continue
            lo, hi = float(s.min()), float(s.max())
            if lo < threshold <= hi:
                rows.append(
                    {
                        "run_id": run_id,
                        "run_label": run_label,
                        "metric": metric,
                        "record_id": record_id,
                        "trial": int(trial),
                        "min_score": lo,
                        "max_score": hi,
                        "threshold": threshold,
                    }
                )
    return pd.DataFrame(rows)


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[str] = config["metrics"]
    runs: dict = config["runs"]
    archive_root = project_root / "output" / "judge_variance_analysis"

    from load_data import load_aggregate_scores_iter, load_scores_iter

    all_scores: list[pd.DataFrame] = []
    all_agg: list[pd.DataFrame] = []
    for run_label, run_cfg in runs.items():
        run_id = run_cfg["run_id"]
        domain = run_cfg.get("domain")
        print(f"  Loading {run_label} ({run_id}, domain={domain}) ...")
        all_scores.append(load_scores_iter(run_id, run_label, archive_root, domain=domain))
        all_agg.append(load_aggregate_scores_iter(run_id, run_label, archive_root, domain=domain))

    scores_df = pd.concat(all_scores, ignore_index=True)
    agg_df = pd.concat(all_agg, ignore_index=True)
    print(f"  {len(scores_df):,} score rows, {len(agg_df):,} aggregate rows loaded")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from eva.metrics.aggregation import EVA_COMPOSITES
    pass_thresholds: dict[str, float] = {
        m: thresh for comp in EVA_COMPOSITES for m, (op, thresh) in comp.thresholds.items() if m in metrics
    }

    print("Computing judge variance ...")
    judge_var = compute_judge_variance(scores_df, metrics)
    print("Computing trial variance ...")
    trial_var = compute_trial_variance(scores_df, metrics)
    print("Computing summaries ...")
    judge_summary = compute_judge_variance_summary(judge_var)
    trial_summary = compute_trial_variance_summary(trial_var)
    print("Computing composite stability ...")
    stability = compute_composite_stability(agg_df)
    print("Computing borderline scenarios ...")
    borderlines = threshold_crossings(scores_df, pass_thresholds)

    # Attach domain to every output by joining on run_label.
    label_to_domain = {label: cfg.get("domain") for label, cfg in runs.items()}
    for out_df in (judge_var, trial_var, judge_summary, trial_summary, stability, borderlines):
        if "run_label" in out_df.columns:
            out_df.insert(
                out_df.columns.get_loc("run_label") + 1,
                "domain",
                out_df["run_label"].map(label_to_domain),
            )

    scores_df.to_csv(output_dir / "scores.csv", index=False)
    judge_var.to_csv(output_dir / "judge_var.csv", index=False)
    trial_var.to_csv(output_dir / "trial_var.csv", index=False)
    judge_summary.to_csv(output_dir / "judge_summary.csv", index=False)
    trial_summary.to_csv(output_dir / "trial_summary.csv", index=False)
    stability.to_csv(output_dir / "composite_stability.csv", index=False)
    borderlines.to_csv(output_dir / "borderline_scenarios.csv", index=False)

    print(f"Wrote 7 CSVs to {output_dir}")


if __name__ == "__main__":
    main()
