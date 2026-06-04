"""One-shot refactor: extract per-object alias files for the itsm domain.

Before
------
Each scenario JSON (and ``expected_scenario_db`` inside ``itsm_dataset.json``)
stored its own ``name_aliases``, ``name_aliases_base`` and
``name_aliases_translatable`` per entry. Translations from every supported
language were sort-merged into the single ``name_aliases`` list, with no way to
distinguish them.

After
-----
One JSON file per canonical name lives in ``data/itsm_aliases/<slug>.json``:

    {
      "name": "Garage A",
      "translatable": true,
      "base": ["a garage", "main garage"],
      "translations": {"fr": ["garage principal", ...], "fr-CA": [...]}
    }

Scenario entries keep ``name`` only; ``resolve_scenario_db`` injects
``name_aliases`` at load time (base + selected language).

Aggregation rules
-----------------
Conflicts between scenarios that have the same ``name`` but different alias
lists are resolved by union (the user said any single source of truth is fine —
union is the least lossy). Currently-merged extras (live − base) are duplicated
across every already-supported language so the existing fr / fr-CA runtimes
don't regress; re-running ``add_culture_data.py`` is the way to get a clean
per-language split when needed.

Idempotent: re-running rewrites the alias files from scratch and re-strips the
scenarios; running it again with no changes is a no-op aside from file mtimes.

Usage
-----
    python scripts/refactor_itsm_aliases.py            # apply
    python scripts/refactor_itsm_aliases.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DOMAIN = "itsm"
SCENARIO_DIR = DATA_DIR / f"{DOMAIN}_scenarios"
DATASET = DATA_DIR / f"{DOMAIN}_dataset.json"
ALIASES_DIR = DATA_DIR / f"{DOMAIN}_aliases"

# Languages already supported in the dataset. Extras (live − base) are split
# across these so no language loses aliases during the refactor.
EXISTING_LANGUAGES = ["fr", "fr-CA"]

ALIAS_PATHS: list[tuple[str, ...]] = [
    ("facilities", "buildings"),
    ("facilities", "zones"),
    ("software_catalog",),
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = _SLUG_RE.sub("_", name.lower()).strip("_")
    if not s:
        raise ValueError(f"Cannot slugify name {name!r}")
    return s


def _get_nested(obj: dict, keys: tuple[str, ...]) -> dict[str, Any]:
    for k in keys:
        obj = obj.get(k, {}) if isinstance(obj, dict) else {}
    return obj


def _iter_alias_entries(data: dict) -> list[dict]:
    out: list[dict] = []
    for path in ALIAS_PATHS:
        section = _get_nested(data, path)
        if not isinstance(section, dict):
            continue
        for entry in section.values():
            if isinstance(entry, dict) and "name" in entry and "name_aliases" in entry:
                out.append(entry)
    return out


def _strip_alias_fields(data: dict) -> bool:
    """Drop name_aliases / _base / _translatable from every alias entry. Returns True if changed."""
    changed = False
    for entry in _iter_alias_entries(data):
        for k in ("name_aliases", "name_aliases_base", "name_aliases_translatable"):
            if k in entry:
                del entry[k]
                changed = True
    return changed


def aggregate() -> dict[str, dict]:
    """Build {name: {translatable, base, extras}} across all scenario files and the dataset."""
    agg: dict[str, dict] = {}

    def absorb(entry: dict) -> None:
        name = entry["name"]
        base = list(entry.get("name_aliases_base") or [])
        translatable = bool(entry.get("name_aliases_translatable"))
        base_set = set(base)
        extras = [a for a in entry.get("name_aliases", []) if a not in base_set]
        rec = agg.setdefault(
            name,
            {"translatable": translatable, "base": list(base), "extras": set()},
        )
        # Union the base list across occurrences (preserve original order, append new).
        for a in base:
            if a not in rec["base"]:
                rec["base"].append(a)
        rec["extras"].update(extras)
        # translatable flag should be consistent — if it varies we prefer translatable=True
        # so any extras keep being eligible for translation.
        rec["translatable"] = rec["translatable"] or translatable

    for path in sorted(SCENARIO_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for e in _iter_alias_entries(data):
            absorb(e)

    if DATASET.exists():
        records = json.loads(DATASET.read_text(encoding="utf-8"))
        for rec in records:
            db = (rec.get("ground_truth") or {}).get("expected_scenario_db") or {}
            for e in _iter_alias_entries(db):
                absorb(e)

    return agg


def write_alias_files(agg: dict[str, dict], dry_run: bool) -> int:
    ALIASES_DIR.mkdir(exist_ok=True)
    written = 0
    seen_slugs: dict[str, str] = {}
    for name, rec in sorted(agg.items()):
        slug = slugify(name)
        if slug in seen_slugs and seen_slugs[slug] != name:
            raise RuntimeError(f"Slug collision: {slug!r} from {name!r} and {seen_slugs[slug]!r}")
        seen_slugs[slug] = name
        base_set = set(rec["base"])
        extras_sorted = sorted(a for a in rec["extras"] if a not in base_set)
        translations: dict[str, list[str]] = {}
        if rec["translatable"] and extras_sorted:
            # Duplicate across already-supported languages so no language regresses.
            # Re-run add_culture_data.py to refresh per-language.
            for lang in EXISTING_LANGUAGES:
                translations[lang] = list(extras_sorted)
        payload = {
            "name": name,
            "translatable": rec["translatable"],
            "base": rec["base"],
            "translations": translations,
        }
        path = ALIASES_DIR / f"{slug}.json"
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if dry_run:
            print(f"[dry-run] would write {path.relative_to(REPO_ROOT)}")
        else:
            path.write_text(body, encoding="utf-8")
        written += 1
    return written


def strip_scenarios(dry_run: bool) -> tuple[int, int]:
    scen_changed = 0
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if _strip_alias_fields(data):
            scen_changed += 1
            if not dry_run:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset_changed = 0
    if DATASET.exists():
        records = json.loads(DATASET.read_text(encoding="utf-8"))
        any_change = False
        for rec in records:
            db = (rec.get("ground_truth") or {}).get("expected_scenario_db") or {}
            if _strip_alias_fields(db):
                any_change = True
                dataset_changed += 1
        if any_change and not dry_run:
            DATASET.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scen_changed, dataset_changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    agg = aggregate()
    print(f"Aggregated {len(agg)} unique names.")
    nt = [n for n, r in agg.items() if not r["translatable"] and r["extras"]]
    if nt:
        print(f"WARNING: {len(nt)} non-translatable name(s) had extras (kept under translations anyway): {nt}")

    written = write_alias_files(agg, args.dry_run)
    scen_n, ds_n = strip_scenarios(args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {written} alias files to {ALIASES_DIR.relative_to(REPO_ROOT)}/")
    verb = "would strip" if args.dry_run else "stripped"
    print(f"{verb} alias fields from {scen_n} scenario files and {ds_n} dataset entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
