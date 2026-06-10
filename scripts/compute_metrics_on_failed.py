"""Compute all registered metrics on a single `_failed_attempt_N` record dir.

The metrics runner skips any record directory matching the `_failed_attempt_N`
suffix. This script bypasses that by renaming the failed dir to the canonical
`<record_id>` form (only if no canonical dir is in the way), then invoking
MetricsRunner targeting that record id.

Usage:
    PYTHONPATH=src python scripts/compute_metrics_on_failed.py \\
        --run-dir output/<run_id> \\
        --failed-dir 2.2.4_failed_attempt_1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import eva.metrics  # noqa: F401  -- triggers metric registration
from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.utils import router


def _extract_user_names(db: dict, record: EvaluationRecord) -> tuple[str, str, str]:
    """Extract user names.

    Walk the resolved scenario_db and return (first_name, last_name, email_local)
    for the entry whose English first+last name keys match the record's English
    culture_overrides — i.e. the user's slot.
    """

    def walk(obj):
        if isinstance(obj, dict):
            if (
                isinstance(obj.get("first_name"), str)
                and isinstance(obj.get("last_name"), str)
                and isinstance(obj.get("email"), str)
            ):
                yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)

    candidates = list(walk(db))
    if not candidates:
        raise ValueError("No first_name/last_name/email triple found in scenario_db.json")
    # If multiple, prefer the one whose email local matches lowercased first.last.
    for c in candidates:
        local = c["email"].split("@", 1)[0]
        if local.lower() == f"{c['first_name'].lower()}.{c['last_name'].lower()}":
            return c["first_name"], c["last_name"], local
    c = candidates[0]
    return c["first_name"], c["last_name"], c["email"].split("@", 1)[0]


def _parse_record_id(failed_name: str) -> str:
    # Strip the trailing "_failed_attempt_<n>" suffix.
    import re

    m = re.match(r"^(.*?)_failed_attempt_\d+$", failed_name)
    if not m:
        raise ValueError(f"{failed_name!r} does not look like a failed-attempt directory")
    return m.group(1)


async def amain(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    records_dir = run_dir / "records"
    failed_dir = records_dir / args.failed_dir
    if not failed_dir.is_dir():
        raise FileNotFoundError(failed_dir)

    record_id = _parse_record_id(args.failed_dir)
    canonical_dir = records_dir / record_id
    if canonical_dir.exists():
        raise FileExistsError(f"{canonical_dir} already exists — move it aside before running this script.")

    config = json.loads((run_dir / "config.json").read_text())
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if "EVA_LANGUAGE" not in os.environ and config.get("language"):
        os.environ["EVA_LANGUAGE"] = config["language"]
        print(f"Set EVA_LANGUAGE={os.environ['EVA_LANGUAGE']} from run config")

    records = EvaluationRecord.load_dataset(dataset_path)
    target = next((r for r in records if r.id == record_id), None)
    if target is None:
        raise ValueError(f"Record id {record_id!r} not found in {dataset_path}")

    print(f"Renaming {failed_dir.name} -> {canonical_dir.name}")
    failed_dir.rename(canonical_dir)

    # If the dataset no longer has culture_overrides for the run's language
    # (e.g. you reverted the file after running), reconstruct it from the
    # resolved scenario_db.json that the worker materialized at run start.
    lang = os.environ.get("EVA_LANGUAGE", "en")
    if lang != "en" and lang not in target.culture_overrides:
        scenario_db_path = canonical_dir / "scenario_db.json"
        if not scenario_db_path.exists():
            raise FileNotFoundError(
                f"culture_overrides[{lang!r}] missing on record and no {scenario_db_path} to recover from"
            )
        db = json.loads(scenario_db_path.read_text())
        first, last, email_local = _extract_user_names(db, target)
        target.culture_overrides[lang] = {"first_name": first, "last_name": last}
        if "." in email_local:
            rfirst, rlast = email_local.split(".", 1)
        else:
            rfirst, rlast = first.lower(), last.lower()
        target.romanized_culture_overrides[lang] = {"first_name": rfirst, "last_name": rlast}
        print(f"Recovered culture_overrides[{lang}]={target.culture_overrides[lang]}")
        print(f"Recovered romanized_culture_overrides[{lang}]={target.romanized_culture_overrides[lang]}")
    if lang != "en" and lang not in target.starting_utterances:
        # Pull from the record's audit log or scenario context if present, else
        # fall back to the English utterance (utterance content rarely affects
        # post-hoc metrics).
        target.starting_utterances[lang] = target.starting_utterances.get("en", "")

    metric_names = args.metrics.split(",") if args.metrics else None
    runner_obj = MetricsRunner(
        run_dir=run_dir,
        dataset=records,
        metric_names=metric_names,
        record_ids=[record_id],
        force_rerun=True,
    )
    result = await runner_obj.run()
    print(f"\nDone. {result.total_records} record(s) evaluated.")
    print(f"Metrics written to: {canonical_dir / 'metrics.json'}")
    return 0


def main() -> int:
    load_dotenv()
    model_list_env = os.getenv("EVA_MODEL_LIST")
    if model_list_env:
        router.init(json.loads(model_list_env))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Path to output/<run_id>")
    ap.add_argument("--failed-dir", required=True, help="Name of the failed_attempt dir under records/")
    ap.add_argument("--metrics", help="Comma-separated metric names (default: all registered)")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
