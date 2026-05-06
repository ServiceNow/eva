"""Run all statistical analyses for eva-bench-stats.

Run from project root:
    uv run python analysis/eva-bench-stats/run_stats.py
    uv run python analysis/eva-bench-stats/run_stats.py --skip-lmm
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stats_CIs import main as _CIs_stats
from stats_frontier import main as _frontier_stats
from stats_perturbations import main as _perturbations_stats
from stats_variance import main as _variance_stats
from stats_variance_lmm import main as _variance_lmm_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all EVA-Bench statistical analyses.")
    parser.add_argument(
        "--skip-lmm",
        action="store_true",
        help="Skip LMM variance decomposition (faster; use when LMM results are already current).",
    )
    args = parser.parse_args()

    print("=== Perturbations: statistics ===")
    _perturbations_stats()
    print()

    print("=== Variance: statistics ===")
    try:
        _variance_stats()
    except FileNotFoundError as e:
        print(f"  [Variance] skipped: {e}")
    print()

    print("=== Variance: LMM decomposition ===")
    if args.skip_lmm:
        print("  [LMM] skipped (--skip-lmm)")
    else:
        try:
            _variance_lmm_stats()
        except FileNotFoundError as e:
            print(f"  [LMM] skipped: {e}")
    print()

    print("=== CIs: statistics ===")
    try:
        _CIs_stats()
    except FileNotFoundError as e:
        print(f"  [CIs] skipped: {e}")
    print()

    print("=== Frontier: statistics ===")
    try:
        _frontier_stats()
    except FileNotFoundError as e:
        print(f"  [Frontier] skipped: {e}")
    print()


if __name__ == "__main__":
    main()
