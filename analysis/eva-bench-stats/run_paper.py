#!/usr/bin/env python3
"""Generate paper-ready LaTeX tables, scatter PDFs, and Pareto-frontier JSON.

Reads results_pooled.csv produced by stats_CIs.py and writes to
output_processed/eva-bench-stats/CIs/paper/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paper_config import load_paper_config  # noqa: E402
from paper_frontier import write_frontier_json  # noqa: E402
from paper_plots import write_scatter  # noqa: E402
from paper_tables import write_accuracy_table, write_experience_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "local" / "eva-bench-stats" / "CIs_config.yaml"
DEFAULT_POOLED = PROJECT_ROOT / "output_processed" / "eva-bench-stats" / "CIs" / "stats" / "results_pooled.csv"

SCATTER_LABELS = {
    "pass_at_1":    {"title": "Accuracy vs Experience pass@1",
                     "x": "Accuracy (EVA-A pass@1)",
                     "y": "Experience (EVA-X pass@1)",
                     "filename": "accuracy_vs_experience_pass_at_1.pdf"},
    "pass_at_k":    {"title": "Accuracy vs Experience pass@k",
                     "x": "Accuracy (EVA-A pass@k)",
                     "y": "Experience (EVA-X pass@k)",
                     "filename": "accuracy_vs_experience_pass_at_k.pdf"},
    "pass_power_k": {"title": "Accuracy vs Experience pass^k",
                     "x": "Accuracy (EVA-A pass^k)",
                     "y": "Experience (EVA-X pass^k)",
                     "filename": "accuracy_vs_experience_pass_power_k.pdf"},
}


def run(config_path: Path, pooled_path: Path) -> None:
    if not pooled_path.exists():
        raise FileNotFoundError(
            f"results_pooled.csv not found at {pooled_path}. Run stats_CIs.py first."
        )
    cfg = load_paper_config(config_path)
    pooled_df = pd.read_csv(pooled_path)
    out_dir = PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")

    accuracy_path = out_dir / "accuracy_table.tex"
    write_accuracy_table(pooled_df, cfg, accuracy_path)
    print(f"  wrote {accuracy_path.name}")

    experience_path = out_dir / "experience_table.tex"
    write_experience_table(pooled_df, cfg, experience_path)
    print(f"  wrote {experience_path.name}")

    for variant_key, info in SCATTER_LABELS.items():
        axes = cfg.scatter.get(variant_key)
        if axes is None:
            print(f"  [skip] no scatter config for '{variant_key}'")
            continue
        scatter_path = out_dir / info["filename"]
        n = write_scatter(
            pooled_df, cfg,
            x_metric=axes["x"], y_metric=axes["y"],
            x_label=info["x"], y_label=info["y"],
            title=info["title"], out_path=scatter_path,
        )
        if n == 0:
            print(f"  [skip] {variant_key}: no rows in pooled CSV")
        else:
            print(f"  wrote {scatter_path.name} ({n} systems)")

    frontier_path = out_dir / "pareto_frontier.json"
    write_frontier_json(pooled_df, cfg, frontier_path)
    print(f"  wrote {frontier_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--pooled", type=Path, default=DEFAULT_POOLED)
    args = ap.parse_args()
    run(args.config, args.pooled)


if __name__ == "__main__":
    main()
