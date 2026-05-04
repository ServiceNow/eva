# Config: local/eva-bench-stats/frontier_config.yaml
#
# trial_scores_path: output/eva-bench-stats/<zip_name>/trial_scores.csv
# output_dir: output_processed/eva-bench-stats/frontier
# random_seed: 42
# n_bootstrap: 1000
# quantiles: [0.75, 0.90]
# alpha: 0.05
#
# models:
#   <display_label>:
#     alias: "<system_alias from trial_scores.csv>"
#     type: cascade  # or s2s

"""Process trial scores for frontier analysis.

Reads clean-baseline rows from trial_scores.csv, computes model-level EVA-A
and EVA-X pass@1 for each configured model, and writes model_scores.csv.

pass@1 for a model = mean across scenarios of (fraction of trials passing per
scenario). This matches the pass_at_1 calculation in data_variance.py.

Run from project root:
    uv run python analysis/eva-bench-stats/data_frontier.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "frontier_config.yaml"


def compute_model_scores(trial_scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute model-level EVA-A and EVA-X pass@1 from clean baseline trial scores.

    pass@1 for a model = mean across scenarios of (fraction of trials passing per
    scenario). Only clean-baseline rows are used; perturbation conditions are excluded.
    """
    alias_to_meta: dict[str, tuple[str, str]] = {
        cfg["alias"]: (label, cfg["type"])
        for label, cfg in config["models"].items()
    }

    clean = trial_scores[
        (trial_scores["perturbation_category"] == "clean")
        & (trial_scores["metric"].isin(["EVA-A_pass", "EVA-X_pass"]))
    ]

    scenario_means = (
        clean.groupby(["system_alias", "domain", "scenario_id", "metric"], sort=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "scenario_pass_rate"})
    )

    model_means = (
        scenario_means.groupby(["system_alias", "metric"], sort=False)["scenario_pass_rate"]
        .mean()
        .reset_index()
        .rename(columns={"scenario_pass_rate": "pass_at_1"})
    )

    wide = (
        model_means.pivot(index="system_alias", columns="metric", values="pass_at_1")
        .reset_index()
        .rename(columns={"EVA-A_pass": "eva_a", "EVA-X_pass": "eva_x"})
    )

    configured_aliases = set(alias_to_meta.keys())
    found_aliases = set(wide["system_alias"])
    missing = configured_aliases - found_aliases
    if missing:
        print(
            f"WARNING: {len(missing)} configured model(s) not found in trial_scores: {sorted(missing)}",
            file=sys.stderr,
        )
        print(
            "Exiting — check trial_scores_path and model aliases in frontier_config.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    for _, row in wide.iterrows():
        alias = row["system_alias"]
        if alias not in alias_to_meta:
            continue
        label, model_type = alias_to_meta[alias]
        rows.append({
            "model_label": label,
            "eva_a": float(row["eva_a"]),
            "eva_x": float(row["eva_x"]),
            "model_type": model_type,
        })

    return pd.DataFrame(rows, columns=["model_label", "eva_a", "eva_x", "model_type"])


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    trial_scores_path = project_root / config["trial_scores_path"]
    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trial scores from {trial_scores_path} ...")
    trial_scores = pd.read_csv(trial_scores_path)
    print(f"  {len(trial_scores):,} rows loaded")

    print("Computing model-level pass@1 scores ...")
    scores = compute_model_scores(trial_scores, config)
    print(f"  {len(scores)} models processed")

    out_path = output_dir / "model_scores.csv"
    scores.to_csv(out_path, index=False)
    print(f"Wrote {len(scores)} rows -> {out_path}")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
