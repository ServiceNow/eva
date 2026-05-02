"""Run all statistical analyses for eva-bench-stats.

Run from project root:
    uv run python analysis/eva-bench-stats/run_stats.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stats_perturbations import main as _perturbations_stats
from stats_variance import main as _variance_stats
from stats_CIs import main as _CIs_stats


def main() -> None:
    print("=== Perturbations: statistics ===")
    _perturbations_stats()
    print()

    print("=== Variance: statistics ===")
    _variance_stats()
    print()

    print("=== CIs: statistics ===")
    try:
        _CIs_stats()
    except FileNotFoundError as e:
        print(f"  [CIs] skipped: {e}")
    print()


if __name__ == "__main__":
    main()
