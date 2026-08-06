#!/usr/bin/env python3
"""Run the perturbation analysis end to end: data → stats.

Reads perturbations_config.yaml and writes results_*.csv to its configured output_dir.
"""

from data_perturbations import main as data_main
from stats_perturbations import main as stats_main

if __name__ == "__main__":
    data_main()
    stats_main()
