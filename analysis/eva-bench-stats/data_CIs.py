"""Process clean run trial-score data into scenario-level means for CI analysis.

Reads trial_scores.csv produced by local/eva-bench-stats/pull_clean_data.py,
filters to configured (alias, perturbation_category=="clean") combinations,
checks completeness against expected scenario counts and trial counts, and
writes scenario-level means.

Run from project root:
    uv run python analysis/eva-bench-stats/data_CIs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "CIs_config.yaml"

# Derived per-scenario metrics computed from binary trial pass values.
# mean over trials gives pass@1 (already produced by compute_scenario_means).
# max over trials gives pass@k for k = expected_k (== 5).
# (mean ** k) gives the plug-in pass^k estimator.
DERIVED_PASS_METRICS: dict[str, tuple[str, str]] = {
    "EVA-A_pass_at_k":    ("EVA-A_pass", "max"),
    "EVA-A_pass_power_k": ("EVA-A_pass", "mean_pow_k"),
    "EVA-X_pass_at_k":    ("EVA-X_pass", "max"),
    "EVA-X_pass_power_k": ("EVA-X_pass", "mean_pow_k"),
}


ALIAS_REMAP: dict[str, str] = {
    # ITSM ultravox runs land under "fixie-ai/ultravox" because the run dir is
    # nested one level deeper than usual; collapse to plain "ultravox" so all
    # three domains share one alias.
    "fixie-ai/ultravox": "ultravox",
}


def load_trial_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trial"] = df["trial"].astype(int)
    df["value"] = df["value"].astype(float)
    df["system_alias"] = df["system_alias"].replace(ALIAS_REMAP)
    return df


def check_completeness(
    df: pd.DataFrame,
    alias: str,
    expected_scenarios: dict[str, int],
    expected_k: int,
    metrics: list[str],
) -> tuple[bool, list[dict]]:
    """Validate per-domain coverage for one model.

    Returns (model_complete, per_domain_rows). per_domain_rows has one entry per
    expected domain with keys: alias, domain, n_scenarios, n_expected,
    n_scenarios_with_wrong_trial_count, n_metrics_missing, complete, issues (str).
    """
    rows: list[dict] = []
    model_df = df[df["system_alias"] == alias]
    sentinel = "task_completion" if "task_completion" in metrics else metrics[0]

    for domain, n_expected in expected_scenarios.items():
        d = model_df[(model_df["domain"] == domain) & (model_df["metric"] == sentinel)]
        n_scenarios = d["scenario_id"].nunique()
        if n_scenarios > 0:
            trial_counts = d.groupby("scenario_id")["trial"].nunique()
            bad_trials = int((trial_counts < expected_k).sum())
        else:
            bad_trials = 0

        # Count metric coverage gaps among the configured metrics.
        # Derived metrics aren't in trial_scores.csv — they're computed downstream.
        # Only check coverage for metrics that are actually expected in trial data.
        source_metrics = [m for m in metrics if m not in DERIVED_PASS_METRICS]
        configured_present = (
            model_df[(model_df["domain"] == domain) & (model_df["metric"].isin(source_metrics))]
            .groupby("metric")["scenario_id"]
            .nunique()
        )
        n_missing_metrics = sum(1 for m in source_metrics if configured_present.get(m, 0) == 0)

        ok = (n_scenarios == n_expected) and (bad_trials == 0) and (n_missing_metrics == 0)
        issues_parts: list[str] = []
        if n_scenarios != n_expected:
            issues_parts.append(f"{n_scenarios}/{n_expected} scenarios")
        if bad_trials > 0:
            issues_parts.append(f"{bad_trials} scenarios with <{expected_k} trials")
        if n_missing_metrics > 0:
            issues_parts.append(f"{n_missing_metrics} metrics absent")

        rows.append({
            "alias": alias,
            "domain": domain,
            "n_scenarios": n_scenarios,
            "n_expected": n_expected,
            "n_scenarios_with_wrong_trial_count": bad_trials,
            "n_metrics_missing": n_missing_metrics,
            "complete": ok,
            "issues": "; ".join(issues_parts),
        })

    model_complete = all(r["complete"] for r in rows)
    return model_complete, rows


def compute_scenario_means(
    df: pd.DataFrame,
    alias: str,
    metrics: list[str],
) -> pd.DataFrame:
    """Mean over trials per (system_alias, domain, scenario_id, metric).

    For binary pass metrics (EVA-A_pass / EVA-X_pass) this mean is the
    scenario-level pass proportion = pass@1.
    """
    sub = df[(df["system_alias"] == alias) & (df["metric"].isin(metrics))]
    if sub.empty:
        return pd.DataFrame(columns=["system_alias", "domain", "scenario_id", "metric", "scenario_mean"])
    keys = ["system_alias", "domain", "scenario_id", "metric"]
    return sub.groupby(keys, sort=False)["value"].mean().reset_index().rename(columns={"value": "scenario_mean"})


def compute_derived_pass_metrics(
    df: pd.DataFrame,
    alias: str,
    derived_metrics: list[str],
    expected_k: int,
) -> pd.DataFrame:
    """Per-scenario pass@k (max over trials) and pass^k (mean^k) for binary pass metrics.

    `derived_metrics` is the subset of the configured metrics list that appears in
    DERIVED_PASS_METRICS. For each derived metric, the source binary pass metric
    (`EVA-A_pass` or `EVA-X_pass`) is read from `df` and reduced per scenario.
    Returns a frame with the same columns as `compute_scenario_means`:
    `system_alias, domain, scenario_id, metric, scenario_mean`.
    """
    out_rows: list[pd.DataFrame] = []
    sub = df[df["system_alias"] == alias]
    keys = ["system_alias", "domain", "scenario_id"]
    for derived_name in derived_metrics:
        if derived_name not in DERIVED_PASS_METRICS:
            continue
        source_metric, reduction = DERIVED_PASS_METRICS[derived_name]
        source = sub[sub["metric"] == source_metric]
        if source.empty:
            continue
        if reduction == "max":
            agg = source.groupby(keys, sort=False)["value"].max().reset_index()
        elif reduction == "mean_pow_k":
            agg = source.groupby(keys, sort=False)["value"].mean().reset_index()
            agg["value"] = agg["value"] ** expected_k
        else:
            raise ValueError(f"Unknown reduction '{reduction}' for {derived_name}")
        agg = agg.rename(columns={"value": "scenario_mean"})
        agg["metric"] = derived_name
        agg = agg[["system_alias", "domain", "scenario_id", "metric", "scenario_mean"]]
        out_rows.append(agg)
    if not out_rows:
        return pd.DataFrame(columns=["system_alias", "domain", "scenario_id", "metric", "scenario_mean"])
    return pd.concat(out_rows, ignore_index=True)


def _resolve_trial_scores_path(config: dict, project_root: Path) -> Path:
    if "trial_scores_dir" in config:
        data_dir = project_root / config["trial_scores_dir"]
        subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("eva_clean_data_"))
        if not subdirs:
            raise FileNotFoundError(f"No eva_clean_data_* subdirectories in {data_dir}")
        path = subdirs[-1] / "trial_scores.csv"
        print(f"Auto-selected most recent clean data folder: {subdirs[-1].name}")
        return path
    return project_root / config["trial_scores_path"]


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    trial_scores_path = _resolve_trial_scores_path(config, project_root)
    output_dir = project_root / config["output_dir"]
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[str] = config["metrics"]
    expected_scenarios: dict[str, int] = config["expected_scenarios"]
    expected_k: int = config["expected_k"]

    print(f"Loading trial scores from {trial_scores_path} ...")
    trial_scores = load_trial_scores(trial_scores_path)
    print(f"  {len(trial_scores):,} rows loaded")

    # Restrict to clean only (puller already filters; double-defense here in case path is overridden).
    trial_scores = trial_scores[trial_scores["perturbation_category"] == "clean"]

    all_means: list[pd.DataFrame] = []
    completeness_rows: list[dict] = []

    for model_label, model_cfg in (config.get("models") or {}).items():
        alias: str = model_cfg["alias"]
        complete, dom_rows = check_completeness(
            trial_scores, alias, expected_scenarios, expected_k, metrics,
        )
        for r in dom_rows:
            completeness_rows.append({"model_label": model_label, "model_complete": complete, **r})
        status = "COMPLETE" if complete else "INCOMPLETE"
        print(f"  [{status}] {model_label}")
        if not complete:
            for r in dom_rows:
                if not r["complete"]:
                    print(f"    - {r['domain']}: {r['issues']}")

        derived_metric_names = [m for m in metrics if m in DERIVED_PASS_METRICS]

        if complete:
            means = compute_scenario_means(trial_scores, alias, metrics)
            if derived_metric_names:
                derived = compute_derived_pass_metrics(
                    trial_scores, alias, derived_metric_names, expected_k
                )
                means = pd.concat([means, derived], ignore_index=True)
            means.insert(0, "model_label", model_label)
            all_means.append(means)
        else:
            # Still include partial scenario means for the domains that ARE complete.
            partial_alias_df = trial_scores[trial_scores["system_alias"] == alias]
            complete_domains = [r["domain"] for r in dom_rows if r["complete"]]
            if complete_domains:
                partial = partial_alias_df[partial_alias_df["domain"].isin(complete_domains)]
                means = compute_scenario_means(partial, alias, metrics)
                if derived_metric_names:
                    derived = compute_derived_pass_metrics(
                        partial, alias, derived_metric_names, expected_k
                    )
                    means = pd.concat([means, derived], ignore_index=True)
                if not means.empty:
                    means.insert(0, "model_label", model_label)
                    all_means.append(means)

    means_df = pd.concat(all_means, ignore_index=True) if all_means else pd.DataFrame(
        columns=["model_label", "system_alias", "domain", "scenario_id", "metric", "scenario_mean"]
    )
    completeness_df = pd.DataFrame(completeness_rows)

    means_path = data_dir / "scenario_means.csv"
    report_path = data_dir / "completeness_report.csv"
    means_df.to_csv(means_path, index=False)
    completeness_df.to_csv(report_path, index=False)

    print(f"\nWrote {len(means_df):,} scenario-mean rows → {means_path}")
    print(f"Wrote completeness report → {report_path}")


if __name__ == "__main__":
    main()
