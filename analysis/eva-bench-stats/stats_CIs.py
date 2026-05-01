# Config: local/eva-bench-stats/CIs_config.yaml
#
# output_dir: output_processed/eva-bench-stats/CIs
# random_seed: 42
# (full config shape TBD)

"""Cluster bootstrap confidence intervals on main scores. Placeholder."""

import pandas as pd


def compute_cis(scores_df: pd.DataFrame, config: dict) -> dict:
    raise NotImplementedError("CI computation not yet implemented.")


if __name__ == "__main__":
    raise NotImplementedError("Standalone run not yet implemented.")
