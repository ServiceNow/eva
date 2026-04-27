"""Test that _current_date is consistent across all representations of a record.

For every record in every dataset:
1. The initial scenario DB and expected final DB must have the same _current_date.
2. The date portion of current_date_time in the dataset record must match
   _current_date in both scenario DBs.

Run with:
    pytest tests/unit/test_current_date_consistency.py -v
"""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Discover all (dataset, scenarios_dir) pairs
DOMAIN_CONFIGS = [
    ("airline_dataset.jsonl", "airline_scenarios"),
    ("medical_hr_dataset.jsonl", "medical_hr_scenarios"),
    ("itsm_dataset.jsonl", "itsm_scenarios"),
]


def _load_records():
    """Yield (domain, record_id, current_date_time, initial_db, expected_db)."""
    for dataset_file, scenarios_dir in DOMAIN_CONFIGS:
        dataset_path = DATA_DIR / dataset_file
        scenarios_path = DATA_DIR / scenarios_dir
        if not dataset_path.exists():
            continue

        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_id = record["id"]
                current_date_time = record.get("current_date_time", "")
                expected_db = record.get("ground_truth", {}).get("expected_scenario_db", {})

                initial_db_path = scenarios_path / f"{record_id}.json"
                if initial_db_path.exists():
                    with open(initial_db_path) as sf:
                        initial_db = json.load(sf)
                else:
                    initial_db = {}

                domain = dataset_file.replace("_dataset.jsonl", "")
                yield domain, record_id, current_date_time, initial_db, expected_db


_ALL_RECORDS = list(_load_records())


@pytest.mark.parametrize(
    "domain,record_id,current_date_time,initial_db,expected_db",
    _ALL_RECORDS,
    ids=[f"{domain}/{rid}" for domain, rid, *_ in _ALL_RECORDS],
)
def test_current_date_matches_between_dbs(domain, record_id, current_date_time, initial_db, expected_db):
    """Initial and expected final scenario DBs must have the same _current_date."""
    initial_date = initial_db.get("_current_date")
    expected_date = expected_db.get("_current_date")

    assert initial_date is not None, f"[{domain}/{record_id}] Initial scenario DB missing _current_date"
    assert expected_date is not None, f"[{domain}/{record_id}] Expected scenario DB missing _current_date"
    assert initial_date == expected_date, (
        f"[{domain}/{record_id}] _current_date mismatch: initial={initial_date!r}, expected_final={expected_date!r}"
    )


@pytest.mark.parametrize(
    "domain,record_id,current_date_time,initial_db,expected_db",
    _ALL_RECORDS,
    ids=[f"{domain}/{rid}" for domain, rid, *_ in _ALL_RECORDS],
)
def test_current_date_time_matches_db(domain, record_id, current_date_time, initial_db, expected_db):
    """The date part of current_date_time must match _current_date in the scenario DBs."""
    assert current_date_time, f"[{domain}/{record_id}] Record missing current_date_time"
    # current_date_time is formatted like "2026-03-17 10:45 CST" — extract the date part
    date_part = current_date_time.split(" ")[0]

    initial_date = initial_db.get("_current_date")
    expected_date = expected_db.get("_current_date")

    if initial_date is not None:
        assert date_part == initial_date, (
            f"[{domain}/{record_id}] current_date_time date ({date_part}) does not match "
            f"initial DB _current_date ({initial_date})"
        )
    if expected_date is not None:
        assert date_part == expected_date, (
            f"[{domain}/{record_id}] current_date_time date ({date_part}) does not match "
            f"expected DB _current_date ({expected_date})"
        )
