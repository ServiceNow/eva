"""One-time migration: tag name_aliases entries as translatable and freeze a base copy.

For every scenario JSON under ``data/<domain>_scenarios/``:
  - Finds all entries that have a ``name_aliases`` field.
  - Calls an LLM once (batched) to classify each unique canonical ``name`` as
    translatable (descriptive English phrases) or not (brand/product proper nouns).
  - Writes ``name_aliases_translatable: true/false`` on each entry.
  - Copies ``name_aliases`` → ``name_aliases_base`` (frozen reference for future
    translations).
  - Leaves ``name_aliases`` unchanged — it is the live list that will grow with
    multilingual entries over time.

Idempotent: entries that already have ``name_aliases_base`` are skipped.

Usage:
  python scripts/migrate_aliases.py --domain itsm
  python scripts/migrate_aliases.py --domain itsm --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eva.utils.json_utils import extract_and_load_json
from eva.utils.llm_client import LLMClient
from eva.utils.logging import get_logger, setup_logging
from eva.utils.router import init

setup_logging()
logger = get_logger(__name__)
load_dotenv()
init(json.loads(os.getenv("EVA_MODEL_LIST")))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL = "gpt-5.2"

# Paths within a scenario JSON that may contain name_aliases entries.
# Each is a tuple of keys to traverse to reach a dict of {code: entry}.
_ALIAS_PATHS: list[tuple[str, ...]] = [
    ("facilities", "buildings"),
    ("facilities", "zones"),
    ("software_catalog",),
]


def _get_nested(obj: dict, keys: tuple[str, ...]) -> dict[str, Any]:
    for k in keys:
        obj = obj.get(k, {})
    return obj


def _iter_alias_entries(data: dict) -> list[tuple[tuple[str, ...], str, dict]]:
    """Yield (path, entry_key, entry) for every entry that has name_aliases."""
    results = []
    for path in _ALIAS_PATHS:
        section = _get_nested(data, path)
        for entry_key, entry in section.items():
            if "name_aliases" in entry:
                results.append((path, entry_key, entry))
    return results


async def _tag_names(names: list[str], llm: LLMClient) -> dict[str, bool]:
    """Ask the LLM to classify each canonical name as translatable or not.

    Translatable = descriptive English phrase (e.g. "East Campus", "North Garage",
    "Engineering Core Access").
    Not translatable = brand/product proper noun (e.g. "Slack", "JetBrains IntelliJ IDEA",
    "Adobe Creative Cloud").
    """
    numbered = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
    prompt = (
        "Classify each item below as translatable or not for a multilingual voice assistant dataset.\n\n"
        "TRANSLATABLE = descriptive English phrases that a non-English speaker would naturally say "
        "in their own language. Examples: building names ('East Campus', 'North Garage'), "
        "department names ('Engineering Core Access'), generic descriptors ('Standard VPN Access').\n\n"
        "NOT TRANSLATABLE = brand names, product names, or proper nouns that are universally "
        "recognised in their original form regardless of language. Examples: 'Slack', "
        "'JetBrains IntelliJ IDEA', 'Adobe Creative Cloud', 'MacBook Pro'.\n\n"
        "When in doubt (e.g. 'Creative Cloud' is borderline), mark as translatable.\n\n"
        'Return JSON: {"results": [{"name": "...", "translatable": true/false}, ...]}\n'
        "Preserve the exact order of the input.\n\n"
        f"Items:\n{numbered}"
    )
    text, _ = await llm.generate_text(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = extract_and_load_json(text)
    results = data.get("results")
    if not isinstance(results, list) or len(results) != len(names):
        raise ValueError(f"Expected {len(names)} classification results, got: {results!r}")
    return {item["name"]: bool(item["translatable"]) for item in results}


def _scenario_files(domain: str) -> list[Path]:
    scenario_dir = DATA_DIR / f"{domain}_scenarios"
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    return sorted(scenario_dir.glob("*.json"))


async def migrate(domain: str, llm: LLMClient, dry_run: bool) -> None:
    files = _scenario_files(domain)
    logger.info(f"Found {len(files)} scenario files for domain={domain!r}")

    # Collect all unique canonical names that need classification
    # from both scenario files and the dataset JSONL.
    unique_names: set[str] = set()
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for _, _, entry in _iter_alias_entries(data):
            if "name_aliases_base" not in entry:
                unique_names.add(entry["name"])

    dataset_path = DATA_DIR / f"{domain}_dataset.json"
    if dataset_path.exists():
        with dataset_path.open(encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            db = rec.get("ground_truth", {}).get("expected_scenario_db")
            if not db:
                continue
            for _, _, entry in _iter_alias_entries(db):
                if "name_aliases_base" not in entry:
                    unique_names.add(entry["name"])

    if not unique_names:
        logger.info("All entries already migrated — nothing to do.")
        return

    names_list = sorted(unique_names)
    logger.info(f"Classifying {len(names_list)} unique names via LLM")
    translatable_map = await _tag_names(names_list, llm)
    logger.info(f"Translatable: {[n for n, t in translatable_map.items() if t]}")
    logger.info(f"Not translatable: {[n for n, t in translatable_map.items() if not t]}")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = False

        for _, _, entry in _iter_alias_entries(data):
            if "name_aliases_base" in entry:
                continue  # already migrated
            name = entry["name"]
            is_translatable = translatable_map.get(name, False)
            entry["name_aliases_translatable"] = is_translatable
            lowercased = [a.lower().strip() for a in entry["name_aliases"]]
            entry["name_aliases"] = lowercased
            entry["name_aliases_base"] = lowercased
            changed = True

        if not changed:
            continue

        if dry_run:
            logger.info(f"[dry-run] would update {path.name}")
        else:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            logger.info(f"Updated {path.name}")

    # Also update expected_scenario_db inside the dataset JSONL.
    if dataset_path.exists():
        records: list[dict] = []
        dataset_changed = False
        with dataset_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        for rec in records:
            db = rec.get("ground_truth", {}).get("expected_scenario_db")
            if not db:
                continue
            for _, _, entry in _iter_alias_entries(db):
                if "name_aliases_base" in entry:
                    continue
                name = entry["name"]
                is_translatable = translatable_map.get(name, False)
                entry["name_aliases_translatable"] = is_translatable
                lowercased = [a.lower().strip() for a in entry["name_aliases"]]
                entry["name_aliases"] = lowercased
                entry["name_aliases_base"] = lowercased
                dataset_changed = True

        if dataset_changed:
            if dry_run:
                logger.info(f"[dry-run] would update {dataset_path.name}")
            else:
                tmp = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.replace(dataset_path)
                logger.info(f"Updated {dataset_path.name}")


async def amain(args: argparse.Namespace) -> int:
    llm = LLMClient(model=args.llm_model, params={"temperature": 0.0})
    domains = args.domains or [
        p.name.removesuffix("_scenarios")
        for p in sorted(DATA_DIR.iterdir())
        if p.is_dir() and p.name.endswith("_scenarios")
    ]
    for domain in domains:
        logger.info(f"=== Domain: {domain} ===")
        await migrate(domain, llm, args.dry_run)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", dest="domains", action="append", help="Domain (repeatable). Default: all.")
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
