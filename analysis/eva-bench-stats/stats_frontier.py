# Config: local/eva-bench-stats/frontier_config.yaml
#
# output_dir: output_processed/eva-bench-stats/frontier
# random_seed: 42
# (full config shape TBD)

"""Frontier analysis: quantile regression, SFA/DEA. Placeholder."""

import pandas as pd


def compute_frontier_stats(scores_df: pd.DataFrame, config: dict) -> dict:
    raise NotImplementedError("Frontier stats not yet implemented.")


if __name__ == "__main__":
    raise NotImplementedError("Standalone run not yet implemented.")
