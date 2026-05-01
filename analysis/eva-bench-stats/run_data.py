"""Run all data preprocessing scripts for eva-bench-stats.

Run from project root:
    uv run python analysis/eva-bench-stats/run_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_perturbations import main as _perturbations_data


def main() -> None:
    print("=== Perturbations: data ===")
    _perturbations_data()
    print()


if __name__ == "__main__":
    main()
