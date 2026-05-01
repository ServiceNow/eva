# Config: local/eva-bench-stats/perturbations_config.yaml
#
# trial_scores_path: output/eva-bench-stats/trial_scores.csv
# output_dir: output_processed/eva-bench-stats/perturbations
# random_seed: 42
# metrics:
#   - EVA-A_mean
#   - EVA-A_pass
#   - EVA-A_pass@3
#   - EVA-A_pass^3
#   - EVA-X_mean
#   - EVA-X_pass
#   - EVA-X_pass@3
#   - EVA-X_pass^3
#   - EVA-overall_mean
#   - task_completion
#   - faithfulness
#   - agent_speech_fidelity
#   - conversation_progression
#   - turn_taking
#   - conciseness
# alpha: 0.05
# n_permutations: 10000
# n_bootstrap: 1000
#
# pass_derivations:   # optional; derive binary scenario-level pass metrics
#   EVA-A_pass@3:     # new metric name
#     source: EVA-A_pass   # per-trial binary column in trial_scores
#     agg: max             # max → any trial passes (pass@k); min → all pass (pass^k)
#   EVA-A_pass^3:
#     source: EVA-A_pass
#     agg: min
#
# models:
#   <display_label>:
#     alias: "<system_alias from trial_scores.csv>"
#     conditions:
#       A: accent
#       B: background_noise
#       "A+B": both

"""Process perturbation trial data into scenario-level delta tables.

Reads trial_scores.csv produced by local/eva-bench-stats/pull_perturbation_data.py,
computes scenario-level means and baseline-vs-perturbation deltas, and writes
processed CSVs to output_processed/eva-bench-stats/perturbations/.

Run from project root:
    uv run python analysis/eva-bench-stats/data_perturbations.py
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "perturbations_config.yaml"


def load_trial_scores(path: Path) -> pd.DataFrame:
    """Load trial_scores.csv and return it with expected column types."""
    df = pd.read_csv(path)
    df["trial"] = df["trial"].astype(int)
    df["value"] = df["value"].astype(float)
    return df


def compute_scenario_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean score across trials for each (alias, domain, condition, scenario, metric).

    Returns DataFrame with columns:
        system_alias, domain, perturbation_category, scenario_id, metric, mean_value
    """
    group_keys = ["system_alias", "domain", "perturbation_category", "scenario_id", "metric"]
    means = df.groupby(group_keys, sort=False)["value"].mean().reset_index().rename(columns={"value": "mean_value"})
    return means


def compute_deltas(
    means_df: pd.DataFrame,
    alias: str,
    condition_map: dict[str, str],
) -> pd.DataFrame:
    """Pair perturbation scenario means with clean baseline and compute deltas.

    Args:
        means_df: Output of compute_scenario_means (may span multiple aliases).
        alias: system_alias string to filter on.
        condition_map: Maps display label → perturbation_category string.
            e.g. {'A': 'accent', 'B': 'background_noise', 'A+B': 'both'}

    Returns DataFrame with columns:
        system_alias, domain, perturbation_condition, scenario_id, metric,
        baseline_mean, perturb_mean, delta
    """
    model_means = means_df[means_df["system_alias"] == alias]

    baseline = model_means[model_means["perturbation_category"] == "clean"][
        ["system_alias", "domain", "scenario_id", "metric", "mean_value"]
    ].rename(columns={"mean_value": "baseline_mean"})

    rows: list[pd.DataFrame] = []
    join_keys = ["system_alias", "domain", "scenario_id", "metric"]

    for label, pert_cat in condition_map.items():
        perturb = model_means[model_means["perturbation_category"] == pert_cat][
            ["system_alias", "domain", "scenario_id", "metric", "mean_value"]
        ].rename(columns={"mean_value": "perturb_mean"})

        merged = baseline.merge(perturb, on=join_keys, how="inner")
        merged["perturbation_condition"] = label
        merged["delta"] = merged["perturb_mean"] - merged["baseline_mean"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "system_alias",
                "domain",
                "perturbation_condition",
                "scenario_id",
                "metric",
                "baseline_mean",
                "perturb_mean",
                "delta",
            ]
        )

    result = pd.concat(rows, ignore_index=True)
    return result[
        [
            "system_alias",
            "domain",
            "perturbation_condition",
            "scenario_id",
            "metric",
            "baseline_mean",
            "perturb_mean",
            "delta",
        ]
    ]


def check_model_completeness(
    df: pd.DataFrame,
    alias: str,
    condition_map: dict[str, str],
    expected_domains: list[str],
    expected_scenarios: int = 30,
    expected_pert_trials: int = 3,
) -> tuple[bool, dict]:
    """Check whether a model's data is complete enough to include in the analysis.

    Completeness criteria (all must hold):
    - All expected_domains are present for every configured condition
    - Each (domain, condition) has exactly expected_scenarios unique scenarios
    - Each perturbation scenario has exactly expected_pert_trials trials

    Uses EVA-A_mean as the representative metric for counting (it is always
    present in aggregate_metrics and not subject to judge errors).

    Args:
        df: trial_scores filtered to this alias only.
        alias: system_alias (for reporting).
        condition_map: Maps condition label → perturbation_category string.
        expected_domains: List of domain strings that must all be present.
        expected_scenarios: Required unique scenario count per (domain, condition).
        expected_pert_trials: Required trial count per perturbation scenario.

    Returns:
        (is_complete, report) where report has keys:
            is_complete (bool), issues (list[str]),
            condition_coverage (dict: condition_label → domain → {n_scenarios, n_expected, complete})
    """
    sentinel = "EVA-A_mean"
    probe = df[df["metric"] == sentinel]

    issues: list[str] = []
    coverage: dict[str, dict] = {}

    for label, pert_cat in condition_map.items():
        coverage[label] = {}
        pert_probe = probe[probe["perturbation_category"] == pert_cat]
        clean_probe = probe[probe["perturbation_category"] == "clean"]

        for domain in expected_domains:
            pert_d = pert_probe[pert_probe["domain"] == domain]
            clean_d = clean_probe[clean_probe["domain"] == domain]

            n_scenarios = pert_d["scenario_id"].nunique()
            n_expected = expected_scenarios

            # Check trial counts: each scenario should have exactly expected_pert_trials
            if n_scenarios > 0:
                trial_counts = pert_d.groupby("scenario_id")["trial"].nunique()
                bad_trials = int((trial_counts < expected_pert_trials).sum())
            else:
                bad_trials = 0

            # Check clean baseline covers all perturbation scenarios
            pert_scenarios = set(pert_d["scenario_id"].unique())
            clean_scenarios = set(clean_d["scenario_id"].unique())
            missing_clean = len(pert_scenarios - clean_scenarios)

            ok = n_scenarios == n_expected and bad_trials == 0 and missing_clean == 0
            coverage[label][domain] = {
                "n_scenarios": n_scenarios,
                "n_expected": n_expected,
                "n_scenarios_with_wrong_trial_count": bad_trials,
                "n_scenarios_missing_clean_baseline": missing_clean,
                "complete": ok,
            }

            if n_scenarios == 0:
                issues.append(f"condition '{label}' ({pert_cat}) missing in domain '{domain}'")
            elif n_scenarios != n_expected:
                issues.append(f"condition '{label}' domain '{domain}': {n_scenarios}/{n_expected} scenarios")
            if bad_trials > 0:
                issues.append(
                    f"condition '{label}' domain '{domain}': {bad_trials} scenarios with <{expected_pert_trials} trials"
                )
            if missing_clean > 0:
                issues.append(
                    f"condition '{label}' domain '{domain}': "
                    f"{missing_clean} perturbation scenarios missing clean baseline"
                )

    is_complete = len(issues) == 0
    return is_complete, {
        "is_complete": is_complete,
        "issues": issues,
        "condition_coverage": coverage,
    }


def derive_pass_metrics(
    df: pd.DataFrame,
    derivations: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Derive binary pass@k / pass^k scenario metrics from per-trial binary values.

    For each derivation, aggregates per-trial binary values (0/1) within each
    (system_alias, system_type, domain, perturbation_category, scenario_id) group
    using max ("any trial passes → pass@k = 1") or min ("all trials pass → pass^k = 1").
    The result is a single row per scenario per (alias, domain, condition) with the
    aggregated binary value, ready to be appended to trial_scores and processed by
    compute_scenario_means (which will return the aggregated value unchanged since
    there is exactly one row per scenario).

    Args:
        df: trial_scores DataFrame.
        derivations: {new_metric_name: {"source": source_metric, "agg": "max"|"min"}}

    Returns:
        DataFrame in trial_scores format with one row per (alias, domain, condition,
        scenario_id) per derived metric. Empty if any source metric is absent.
    """
    group_keys = ["system_alias", "system_type", "domain", "perturbation_category", "scenario_id"]
    parts: list[pd.DataFrame] = []

    for new_metric, spec in derivations.items():
        source = spec["source"]
        agg = spec["agg"]
        source_rows = df[df["metric"] == source]
        if source_rows.empty:
            continue

        agg_fn = source_rows.groupby(group_keys, sort=False)["value"].agg(agg).reset_index()
        agg_fn["metric"] = new_metric
        agg_fn["trial"] = 0
        agg_fn["run_id"] = ""
        parts.append(agg_fn[list(df.columns)])

    if not parts:
        return pd.DataFrame(columns=list(df.columns))
    return pd.concat(parts, ignore_index=True)


def build_scenario_deltas(
    trial_scores: pd.DataFrame,
    model_label: str,
    alias: str,
    condition_map: dict[str, str],
    metrics: list[str],
) -> pd.DataFrame:
    """Full pipeline: filter → means → deltas for one model.

    Args:
        trial_scores: Full trial_scores.csv as DataFrame.
        model_label: Display label for this model (added as a column).
        alias: system_alias string identifying this model in the data.
        condition_map: Maps condition label → perturbation_category.
        metrics: Which metrics to include. Rows with other metrics are dropped.

    Returns DataFrame with columns:
        model_label, system_alias, domain, perturbation_condition, scenario_id,
        metric, baseline_mean, perturb_mean, delta
    """
    filtered = trial_scores[(trial_scores["system_alias"] == alias) & (trial_scores["metric"].isin(metrics))]

    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "model_label",
                "system_alias",
                "domain",
                "perturbation_condition",
                "scenario_id",
                "metric",
                "baseline_mean",
                "perturb_mean",
                "delta",
            ]
        )

    means = compute_scenario_means(filtered)
    deltas = compute_deltas(means, alias=alias, condition_map=condition_map)

    if deltas.empty:
        return pd.DataFrame(
            columns=[
                "model_label",
                "system_alias",
                "domain",
                "perturbation_condition",
                "scenario_id",
                "metric",
                "baseline_mean",
                "perturb_mean",
                "delta",
            ]
        )

    deltas.insert(0, "model_label", model_label)
    return deltas


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    trial_scores_path = project_root / config["trial_scores_path"]
    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[str] = config["metrics"]
    expected_domains: list[str] = config.get("expected_domains", ["itsm", "medical_hr", "airline"])
    expected_scenarios: int = config.get("expected_scenarios", 30)
    expected_pert_trials: int = config.get("expected_pert_trials", 3)

    pass_derivations: dict[str, dict[str, str]] = config.get("pass_derivations", {})

    print(f"Loading trial scores from {trial_scores_path} ...")
    trial_scores = load_trial_scores(trial_scores_path)
    print(f"  {len(trial_scores):,} rows loaded")

    if pass_derivations:
        derived = derive_pass_metrics(trial_scores, pass_derivations)
        trial_scores = pd.concat([trial_scores, derived], ignore_index=True)
        print(f"  {len(derived):,} derived rows added ({', '.join(pass_derivations)})")

    all_deltas: list[pd.DataFrame] = []
    completeness_rows: list[dict] = []

    for model_label, model_cfg in config["models"].items():
        alias: str = model_cfg["alias"]
        condition_map: dict[str, str] = model_cfg["conditions"]

        model_df = trial_scores[trial_scores["system_alias"] == alias]
        is_complete, report = check_model_completeness(
            model_df,
            alias,
            condition_map,
            expected_domains=expected_domains,
            expected_scenarios=expected_scenarios,
            expected_pert_trials=expected_pert_trials,
        )

        # Flatten coverage into one report row per (model, condition, domain)
        for cond_label, domain_map in report["condition_coverage"].items():
            for domain, info in domain_map.items():
                completeness_rows.append(
                    {
                        "model_label": model_label,
                        "alias": alias,
                        "condition_label": cond_label,
                        "perturbation_category": condition_map[cond_label],
                        "domain": domain,
                        "n_scenarios": info["n_scenarios"],
                        "n_expected": info["n_expected"],
                        "n_scenarios_with_wrong_trial_count": info["n_scenarios_with_wrong_trial_count"],
                        "complete": info["complete"],
                        "model_complete": is_complete,
                        "issues": "; ".join(report["issues"]) if not is_complete else "",
                    }
                )

        status = "COMPLETE" if is_complete else "INCOMPLETE"
        print(f"  [{status}] {model_label}")
        if not is_complete:
            for issue in report["issues"]:
                print(f"    - {issue}")

        if is_complete:
            deltas = build_scenario_deltas(
                trial_scores=trial_scores,
                model_label=model_label,
                alias=alias,
                condition_map=condition_map,
                metrics=metrics,
            )
            print(f"    {len(deltas):,} delta rows")
            all_deltas.append(deltas)

    combined = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()
    completeness_df = pd.DataFrame(completeness_rows)

    deltas_path = output_dir / "scenario_deltas.csv"
    report_path = output_dir / "completeness_report.csv"

    combined.to_csv(deltas_path, index=False)
    completeness_df.to_csv(report_path, index=False)

    n_complete = (
        completeness_df["model_complete"].any()
        and completeness_df.groupby("model_label")["model_complete"].first().sum()
    )
    n_total = len(config["models"])
    print(f"\n{n_complete}/{n_total} models complete and included in analysis")
    print(f"Wrote {len(combined):,} delta rows → {deltas_path}")
    print(f"Wrote completeness report → {report_path}")


if __name__ == "__main__":
    main()
