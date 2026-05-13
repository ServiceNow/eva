#!/usr/bin/env python3
"""Regenerate tests/fixtures/metric_signatures.json.

Run this after intentionally changing a metric's logic and bumping its
`version` class attribute (or after editing its judge prompt template).
The drift test (tests/unit/metrics/test_metric_signatures.py) compares
the current state against this fixture and fails on any unintended drift.

Usage:
    python scripts/regen_metric_signatures.py
"""

import json
from pathlib import Path

from eva.metrics.signatures import compute_all_metric_signatures

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "metric_signatures.json"


def main() -> None:
    signatures = compute_all_metric_signatures()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(signatures, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(signatures)} metric signatures to {FIXTURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
