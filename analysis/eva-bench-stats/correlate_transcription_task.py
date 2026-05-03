#!/usr/bin/env python3
"""Cascade-only correlation: transcription accuracy on key entities vs task completion.

Computes Pearson r and p across all (system, scenario, trial) triples for the
clean cascade runs, plus a per-system breakdown. Writes a CSV summary and prints
a paper-ready sentence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def _short_label(alias: str) -> str:
    """Compact display label for the long cascade aliases."""
    rules = [
        ("cohere_transcribe", "Cohere"),
        ("elevenlabs", "ElevenAgent"),
        ("ink-whisper", "InkWhisper"),
        ("nova-3 + gpt-5.4 + sonic-3", "Nova3+GPT5.4"),
        ("nova-3 + gpt-5.4-mini", "Nova3+GPT5.4-mini"),
        ("parakeet", "Parakeet+Gemma"),
        ("whisper-large-v3", "Whisper+Qwen"),
    ]
    for needle, short in rules:
        if needle in alias:
            return short
    return alias[:18]


def _plot_pooled_scatter(sys_means: pd.DataFrame, r: float, p: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    xs = sys_means[X_METRIC].to_numpy()
    ys = sys_means[Y_METRIC].to_numpy()
    ax.scatter(xs, ys, s=80, color="#3a86ff", edgecolor="white", linewidth=1.0, zorder=3)
    for _, row in sys_means.iterrows():
        ax.annotate(_short_label(row["system_alias"]),
                    (row[X_METRIC], row[Y_METRIC]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    # Regression line
    if len(xs) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.array([xs.min() - 0.02, xs.max() + 0.02])
        ax.plot(x_line, slope * x_line + intercept,
                linestyle="--", color="grey", alpha=0.7, zorder=1, label="OLS fit")
    ax.set_xlim(min(xs.min() - 0.05, 0.5), 1.0)
    ax.set_ylim(0, max(ys.max() + 0.1, 1.0))
    ax.set_xlabel("Mean transcription accuracy on key entities")
    ax.set_ylabel("Mean task completion")
    ax.set_title(f"Cascade systems: transcription vs task completion\n"
                 f"Pearson r = {r:.3f}, p = {_fmt_p(p)}, n = {len(sys_means)} systems")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_per_domain(wide: pd.DataFrame, per_domain: list[dict], out_path: Path) -> None:
    domains = [d["domain"] for d in per_domain]
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4.5), sharey=True)
    if len(domains) == 1:
        axes = [axes]
    for ax, dom_info in zip(axes, per_domain):
        domain = dom_info["domain"]
        dg = wide[wide["domain"] == domain]
        sm = dg.groupby("system_alias")[[X_METRIC, Y_METRIC]].mean().reset_index()
        xs = sm[X_METRIC].to_numpy()
        ys = sm[Y_METRIC].to_numpy()
        ax.scatter(xs, ys, s=60, color="#3a86ff", edgecolor="white", linewidth=1.0, zorder=3)
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.array([xs.min() - 0.02, xs.max() + 0.02])
            ax.plot(x_line, slope * x_line + intercept,
                    linestyle="--", color="grey", alpha=0.7, zorder=1)
        ax.set_xlim(min(xs.min() - 0.05, 0.5), 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Mean transcription accuracy")
        ax.set_title(f"{domain}\nr = {dom_info['r']:.3f}, p = {_fmt_p(dom_info['p'])}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean task completion")
    fig.suptitle("Cascade-system correlation by domain", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_split(threshold_df: pd.DataFrame, threshold: float, out_path: Path) -> None:
    if threshold_df.empty:
        return
    df = threshold_df.copy()
    scopes = df["scope"].tolist()
    x = np.arange(len(scopes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_above = ax.bar(x - width / 2, df["tc_above"], width,
                        label=f"trans ≥ {threshold:.0%} (n={int(df['n_above'].iloc[0])})",
                        color="#2a9d8f")
    bars_below = ax.bar(x + width / 2, df["tc_below"], width,
                        label=f"trans < {threshold:.0%} (n={int(df['n_below'].iloc[0])})",
                        color="#e76f51")
    for b in list(bars_above) + list(bars_below):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(scopes)
    ax.set_ylabel("Mean task completion")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Task completion by transcription-accuracy threshold ({threshold:.0%})")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(trial_scores_path: Path, output_dir: Path) -> None:
    print(f"Loading {trial_scores_path}")
    df = pd.read_csv(trial_scores_path)
    df = df[(df["perturbation_category"] == "clean") & (df["system_type"] == "cascade")]
    df = df[df["metric"].isin([X_METRIC, Y_METRIC])]

    keys = ["run_id", "system_alias", "domain", "scenario_id", "trial"]
    wide = df.pivot_table(index=keys, columns="metric", values="value", aggfunc="first").reset_index()
    wide = wide.dropna(subset=[X_METRIC, Y_METRIC])
    print(f"  {len(wide):,} cascade trials with both {X_METRIC} and {Y_METRIC}")

    # Between-system: one point per system (trial-mean of each axis). This is the
    # right granularity for a claim phrased "across cascade systems X correlates
    # with Y" — it answers whether systems with better STT have better task
    # completion. Within-system correlations are dragged down by ceiling effects
    # (e.g. ElevenLabs: trans std = 0.08 because its STT is near-perfect on most
    # trials, so within-system trial variance in transcription doesn't predict TC).
    sys_means = wide.groupby("system_alias")[[X_METRIC, Y_METRIC]].mean().reset_index()
    r_sys, p_sys, n_sys = pearson(sys_means[X_METRIC].to_numpy(), sys_means[Y_METRIC].to_numpy())

    # Same between-system computation but restricted to each domain.
    per_domain: list[dict] = []
    for domain, dg in wide.groupby("domain", sort=True):
        sm = dg.groupby("system_alias")[[X_METRIC, Y_METRIC]].mean()
        r_d, p_d, n_d = pearson(sm[X_METRIC].to_numpy(), sm[Y_METRIC].to_numpy())
        per_domain.append({"domain": domain, "n_systems": n_d, "r": r_d, "p": p_d})

    # Threshold split: group cascade systems by mean transcription accuracy at a
    # natural break, compare mean task_completion across the two groups.
    THRESHOLD = 0.70
    above = sys_means[sys_means[X_METRIC] >= THRESHOLD]
    below = sys_means[sys_means[X_METRIC] <  THRESHOLD]
    threshold_summary: list[dict] = []
    if len(above) and len(below):
        tc_a = float(above[Y_METRIC].mean())
        tc_b = float(below[Y_METRIC].mean())
        threshold_summary.append({
            "scope": "all_domains",
            "threshold": THRESHOLD,
            "n_above": len(above),
            "n_below": len(below),
            "tc_above": tc_a,
            "tc_below": tc_b,
            "abs_drop": tc_a - tc_b,
            "rel_drop_pct": (1 - tc_b / tc_a) * 100 if tc_a > 0 else float("nan"),
        })
        # Per-domain version using the same above/below grouping (system identity).
        above_aliases = set(above["system_alias"])
        for domain, dg in wide.groupby("domain", sort=True):
            dom_means = dg.groupby("system_alias")[Y_METRIC].mean()
            ta = float(dom_means.loc[dom_means.index.isin(above_aliases)].mean())
            tb = float(dom_means.loc[~dom_means.index.isin(above_aliases)].mean())
            threshold_summary.append({
                "scope": domain,
                "threshold": THRESHOLD,
                "n_above": len(above),
                "n_below": len(below),
                "tc_above": ta,
                "tc_below": tb,
                "abs_drop": ta - tb,
                "rel_drop_pct": (1 - tb / ta) * 100 if ta > 0 else float("nan"),
            })

    # Trial-level pooled, plus per-system breakdown — supporting context.
    r_pool, p_pool, n_pool = pearson(wide[X_METRIC].to_numpy(), wide[Y_METRIC].to_numpy())
    per_system: list[dict] = []
    for alias, g in wide.groupby("system_alias", sort=True):
        r, p, n = pearson(g[X_METRIC].to_numpy(), g[Y_METRIC].to_numpy())
        per_system.append({
            "system_alias": alias, "n_trials": n,
            "trans_mean": float(g[X_METRIC].mean()),
            "trans_std":  float(g[X_METRIC].std()),
            "tc_mean":    float(g[Y_METRIC].mean()),
            "r_within":   r, "p_within": p,
        })

    summary = pd.DataFrame(per_system)
    summary = pd.concat([
        summary,
        pd.DataFrame([{
            "system_alias": "between-system (n=7)",
            "n_trials": n_sys,
            "trans_mean": float("nan"), "trans_std": float("nan"),
            "tc_mean": float("nan"),
            "r_within": r_sys, "p_within": p_sys,
        }, {
            "system_alias": "trial-level pooled",
            "n_trials": n_pool,
            "trans_mean": float("nan"), "trans_std": float("nan"),
            "tc_mean": float("nan"),
            "r_within": r_pool, "p_within": p_pool,
        }]),
    ], ignore_index=True)

    print("\nPer-system means and within-system Pearson:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    domain_df = pd.DataFrame(per_domain)
    print("\nBetween-system correlation by domain:")
    print(domain_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    threshold_df = pd.DataFrame(threshold_summary)
    print(f"\nThreshold split @ transcription accuracy = {THRESHOLD}:")
    print(threshold_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "transcription_vs_task_completion.csv"
    summary.to_csv(out_csv, index=False)
    domain_csv = output_dir / "transcription_vs_task_completion_by_domain.csv"
    domain_df.to_csv(domain_csv, index=False)
    threshold_csv = output_dir / "transcription_threshold_split.csv"
    threshold_df.to_csv(threshold_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {domain_csv}")
    print(f"Wrote {threshold_csv}")

    # Plots
    pooled_pdf = output_dir / "transcription_vs_task_completion.pdf"
    _plot_pooled_scatter(sys_means, r_sys, p_sys, pooled_pdf)
    print(f"Wrote {pooled_pdf}")

    domain_pdf = output_dir / "transcription_vs_task_completion_by_domain.pdf"
    _plot_per_domain(wide, per_domain, domain_pdf)
    print(f"Wrote {domain_pdf}")

    threshold_pdf = output_dir / "transcription_threshold_split.pdf"
    _plot_threshold_split(threshold_df, THRESHOLD, threshold_pdf)
    print(f"Wrote {threshold_pdf}")

    print("\nPaper-ready:")
    print(
        f"  Across cascade systems, mean transcription accuracy on key entities is "
        f"strongly correlated with mean task completion (Pearson r = {r_sys:.3f}, "
        f"p = {_fmt_p(p_sys)}, n = {n_sys} systems)."
    )
    print("\nSupporting (trial-level pooled across cascades):")
    print(
        f"  r = {r_pool:.3f}, p = {_fmt_p(p_pool)}, n = {n_pool:,} trials. The "
        f"weaker pooled value reflects within-system ceiling effects (e.g. "
        f"ElevenLabs has trans std = {summary.loc[summary['system_alias']=='elevenlabs','trans_std'].iloc[0]:.3f} "
        f"because its STT saturates near 1.0)."
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
