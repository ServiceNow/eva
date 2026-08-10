"""Recompute the `target` field for a judged conciseness dataset to match Tara's convention.

Reverse-engineered from her eva_conciseness_test_set.jsonl (verified exact match, 63/63
records): target = the mode (most common) per-turn rating in judge_1.details.per_turn_ratings,
with ties broken toward the LOWER (worse) value. This is a post-hoc label derived from the
actual judge output — not the generation-intent bucket (which prompt variant produced the
trace) used to build the dataset in the first place.

The original generation-intent value is preserved under `generation_target` for traceability
(so you can still see, e.g., "the wordy-prompt run for this record was intended to hit 1, but
the majority of its turns actually judged as 2").

Usage:
    python scripts/recompute_conciseness_target.py --dataset eva_conciseness_test_set_fr.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def mode_tiebreak_min(ratings: list[int]) -> int:
    """Most common rating; ties broken toward the lower (worse) value."""
    counts = Counter(ratings)
    max_count = max(counts.values())
    tied = [v for v, c in counts.items() if c == max_count]
    return min(tied)


def recompute_target(record: dict[str, Any]) -> int | None:
    ratings = [v for v in (record.get("judge_1", {}).get("details") or {}).get("per_turn_ratings", {}).values() if v is not None]
    if not ratings:
        return None
    return mode_tiebreak_min(ratings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute conciseness `target` per Tara's mode/tie-break-min convention.")
    parser.add_argument("--dataset", required=True, help="Path to the conciseness .jsonl dataset (edited in place)")
    parser.add_argument("--out", default=None, help="Output path (default: overwrite --dataset)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    records = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]

    changed = 0
    skipped_no_ratings = 0
    for record in records:
        new_target = recompute_target(record)
        if new_target is None:
            skipped_no_ratings += 1
            continue
        old_target = record.get("target")
        if "generation_target" not in record:
            record["generation_target"] = old_target
        record["target"] = new_target
        if new_target != old_target:
            changed += 1
            print(f"  id={record.get('id')} original_id={record.get('original_id')}: "
                  f"generation_target={old_target} -> target={new_target}")

    out_path = Path(args.out) if args.out else dataset_path
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    print()
    print(f"Recomputed target for {len(records) - skipped_no_ratings}/{len(records)} record(s). "
          f"{changed} differed from generation_target. {skipped_no_ratings} skipped (no judge_1 ratings).")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
