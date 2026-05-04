"""Run all data preprocessing scripts for eva-bench-stats.

Run from project root:
    uv run python analysis/eva-bench-stats/run_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_perturbations import main as _perturbations_data
from data_variance import main as _variance_data
from data_CIs import main as _CIs_data
from data_frontier import main as _frontier_data


def main() -> None:
    print("=== Perturbations: data ===")
    _perturbations_data()
    print()

    print("=== Variance: data ===")
    _variance_data()
    print()

    print("=== CIs: data ===")
    try:
        _CIs_data()
    except FileNotFoundError as e:
        print(f"  [CIs] skipped: {e}")
    print()

    print("=== Frontier: data ===")
    try:
        _frontier_data()
    except FileNotFoundError as e:
        print(f"  [Frontier] skipped: {e}")
    print()


if __name__ == "__main__":
    main()
