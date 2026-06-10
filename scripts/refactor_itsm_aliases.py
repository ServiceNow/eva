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

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = _SLUG_RE.sub("_", name.lower()).strip("_")
    if not s:
        raise ValueError(f"Cannot slugify name {name!r}")
    return s


def _iter_alias_entries(obj: Any) -> list[dict]:
    """Recursively find every dict containing both ``name`` and ``name_aliases``.

    Path-agnostic: catches entries at any nesting depth (top-level
    ``software_catalog`` as well as ``software_catalog.applications/licenses``).
    """
    out: list[dict] = []
    if isinstance(obj, dict):
        if "name" in obj and "name_aliases" in obj:
            out.append(obj)
        for v in obj.values():
            out.extend(_iter_alias_entries(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_alias_entries(item))
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


def _load_existing_index() -> dict[str, dict]:
    """Load already-written alias files keyed by canonical ``name``."""
    if not ALIASES_DIR.exists():
        return {}
    return {(d := json.loads(p.read_text(encoding="utf-8")))["name"]: d for p in ALIASES_DIR.glob("*.json")}


def aggregate(existing: dict[str, dict]) -> dict[str, dict]:
    """Build {name: {translatable, base, extras}} for names not yet in ``existing``.

    Entries lacking ``name_aliases_base``/``name_aliases_translatable`` (never went
    through the original migrate_aliases.py pass) are treated as non-translatable
    with their inline ``name_aliases`` promoted to ``base``. Those were missed
    because the old path list didn't recurse into ``software_catalog.applications/licenses``.
    """
    agg: dict[str, dict] = {}

    def absorb(entry: dict) -> None:
        name = entry["name"]
        if name in existing:
            return  # already captured in a prior migration pass
        inline = list(entry.get("name_aliases") or [])
        if "name_aliases_base" in entry or "name_aliases_translatable" in entry:
            base = list(entry.get("name_aliases_base") or [])
            translatable = bool(entry.get("name_aliases_translatable"))
            base_set = set(base)
            extras = [a for a in inline if a not in base_set]
        else:
            # Never-migrated entry: inline aliases ARE the base; assume non-translatable
            # (these turned out to be brand/product names: Confluence, Jira, SFDC, ...).
            base = inline
            translatable = False
            extras = []
        rec = agg.setdefault(name, {"translatable": translatable, "base": list(base), "extras": set()})
        for a in base:
            if a not in rec["base"]:
                rec["base"].append(a)
        rec["extras"].update(extras)
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
        # Walk the whole record, not just expected_scenario_db: tool_response payloads
        # under ground_truth.events also carry name_aliases that need stripping. The
        # resolver injects aliases at scenario-DB load only, so any frozen alias list
        # buried in golden tool responses would drift from the runtime-resolved form.
        for rec in records:
            if _strip_alias_fields(rec):
                any_change = True
                dataset_changed += 1
        if any_change and not dry_run:
            DATASET.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scen_changed, dataset_changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing = _load_existing_index()
    print(f"Existing alias files: {len(existing)}")
    agg = aggregate(existing)
    print(f"New unique names to add: {len(agg)}")
    if agg:
        print(f"  → {sorted(agg.keys())}")

    written = write_alias_files(agg, args.dry_run)
    scen_n, ds_n = strip_scenarios(args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {written} new alias files to {ALIASES_DIR.relative_to(REPO_ROOT)}/")
    verb = "would strip" if args.dry_run else "stripped"
    print(f"{verb} alias fields from {scen_n} scenario files and {ds_n} dataset entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
