"""Phase A migration: convert datasets + scenario DBs to the culture-overrides schema.

Approach
--------
For each record and each scenario DB file, the migration treats the JSON as a
string and runs case-sensitive, word-boundary regex replacements for every
form of the user's name found in the data:

- Title-case full name   ``"First Last"``  → ``"<FIRST_NAME> <LAST_NAME>"``
- Title-case first name  ``"First"``       → ``"<FIRST_NAME>"``
- Title-case last name   ``"Last"``        → ``"<LAST_NAME>"``
- Lower-case full        ``"first last"``  → ``"<FIRST_NAME_ROMANIZED> <LAST_NAME_ROMANIZED>"``
- Lower-case first       ``"first"``       → ``"<FIRST_NAME_ROMANIZED>"``
- Lower-case last        ``"last"``        → ``"<LAST_NAME_ROMANIZED>"``
- Phone number           ``"+1234567890"`` → ``"<PHONE>"``

Stringified replacement catches sneaky spots like ``edge_cases`` free text,
``session.last_name`` (which is lowercase), and emails — without the resolver
having to know the schema. Word boundaries (``\\b``) avoid partial-word hits like
``Sam`` munching ``Samurai``.

When migrating a record, ``culture_overrides`` and ``romanized_culture_overrides``
are popped first so the names stored there aren't replaced, then restored after.

Idempotent: re-running is safe because placeholders never contain the literal
names.

Usage:
    python scripts/migrate_to_culture_schema.py            # all data/*_dataset.jsonl
    python scripts/migrate_to_culture_schema.py --dry-run  # report only
    python scripts/migrate_to_culture_schema.py data/airline_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from eva.utils.culture import (
    FIRST_NAME_PLACEHOLDER,
    FIRST_NAME_ROMANIZED_PLACEHOLDER,
    LAST_NAME_PLACEHOLDER,
    LAST_NAME_ROMANIZED_PLACEHOLDER,
    PHONE_PLACEHOLDER,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _split_name(full_name: str) -> tuple[str, str]:
    parts = full_name.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Cannot split name into first/last: {full_name!r}")
    return " ".join(parts[:-1]), parts[-1]


def _word_re(text: str) -> re.Pattern[str]:
    """Case-sensitive whole-word regex for ``text``."""
    return re.compile(rf"\b{re.escape(text)}\b")


def _placeholderize_string(s: str, first: str, last: str) -> str:
    """Apply the six-pass name replacement on a string."""
    first_lower = first.lower()
    last_lower = last.lower()

    # Full names first so the inter-word space stays intact.
    s = _word_re(f"{first} {last}").sub(f"{FIRST_NAME_PLACEHOLDER} {LAST_NAME_PLACEHOLDER}", s)
    s = _word_re(f"{first_lower} {last_lower}").sub(
        f"{FIRST_NAME_ROMANIZED_PLACEHOLDER} {LAST_NAME_ROMANIZED_PLACEHOLDER}", s
    )
    # Individual names.
    s = _word_re(first).sub(FIRST_NAME_PLACEHOLDER, s)
    s = _word_re(last).sub(LAST_NAME_PLACEHOLDER, s)
    s = _word_re(first_lower).sub(FIRST_NAME_ROMANIZED_PLACEHOLDER, s)
    s = _word_re(last_lower).sub(LAST_NAME_ROMANIZED_PLACEHOLDER, s)
    return s


def _placeholderize_obj(obj: dict | list, first: str, last: str) -> dict | list:
    """Stringify, replace, re-parse."""
    return json.loads(_placeholderize_string(json.dumps(obj, ensure_ascii=False), first, last))


def migrate_record(record: dict) -> tuple[bool, str, str]:
    """Mutate ``record`` in place. Returns ``(changed, first, last)``."""
    user_config = record.get("user_config") or {}
    existing = record.get("culture_overrides", {}).get("en")
    if existing:
        first, last = existing["first_name"], existing["last_name"]
    else:
        name = user_config.get("name")
        if not name:
            raise ValueError(f"Record {record.get('id')!r} missing user_config.name")
        # The name field itself may already be placeholderized from a prior run;
        # in that case culture_overrides.en should have existed, so the branch
        # above would have hit. If we got here, name is the raw string.
        first, last = _split_name(name)

    changed = False

    if not existing:
        record.setdefault("culture_overrides", {})["en"] = {"first_name": first, "last_name": last}
        changed = True
    if "en" not in record.get("romanized_culture_overrides", {}):
        record.setdefault("romanized_culture_overrides", {})["en"] = {
            "first_name": first,
            "last_name": last,
        }
        changed = True

    # Move opening utterance out of user_goal to a top-level dict.
    record.setdefault("starting_utterances", {})
    if "starting_utterance" in record["user_goal"]:
        record["starting_utterances"].setdefault("en", record["user_goal"].pop("starting_utterance"))
        changed = True
    if "en_starting_utterance" in record["user_goal"]:
        record["starting_utterances"].setdefault("en", record["user_goal"].pop("en_starting_utterance"))
        changed = True
    for k in list(record["user_goal"].keys()):
        if k.endswith("_starting_utterance"):
            lang = k.removesuffix("_starting_utterance")
            record["starting_utterances"].setdefault(lang, record["user_goal"].pop(k))
            changed = True
    if "en" not in record["starting_utterances"]:
        raise ValueError(f"Record {record.get('id')!r} missing starting_utterance")

    # Now placeholderize the rest of the record. Pop the override dicts so the
    # canonical name values stored there aren't substituted.
    culture = record.pop("culture_overrides")
    romanized = record.pop("romanized_culture_overrides")
    before = json.dumps(record, ensure_ascii=False, sort_keys=True)
    new_record = _placeholderize_obj(record, first, last)
    after = json.dumps(new_record, ensure_ascii=False, sort_keys=True)
    record.clear()
    record.update(new_record)
    record["culture_overrides"] = culture
    record["romanized_culture_overrides"] = romanized
    if _word_re(first).search(after) or _word_re(last).search(after):
        raise RuntimeError(f"Record {record.get('id')!r}: name leaked past placeholderization — debug needed")
    if before != after:
        changed = True
    return changed, first, last


def _looks_like_phone(value: Any) -> bool:
    """Return True if ``value`` looks like a full phone number.

    Accepts values that contain '+' or have 7+ consecutive digits.
    Rejects short strings like phone_last_four (4-digit strings).
    """
    if not isinstance(value, str):
        return False
    digit_count = sum(c.isdigit() for c in value)
    return digit_count >= 7


def _extract_phone(obj: Any) -> str | None:
    """Recursively search ``obj`` for a key named 'phone' with a phone-like value.

    Returns the first matching value found, or None.
    Only looks at keys literally named 'phone' — skips 'phone_last_four' etc.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "phone" and isinstance(v, str) and _looks_like_phone(v):
                return v
        for v in obj.values():
            result = _extract_phone(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _extract_phone(item)
            if result is not None:
                return result
    return None


def migrate_scenario_db(path: Path, first: str, last: str, dry_run: bool) -> tuple[bool, str | None]:
    """Placeholderize names and phone numbers in a scenario DB file.

    Returns ``(changed, phone_value_or_None)``.
    """
    if not path.exists():
        return False, None
    original = path.read_text(encoding="utf-8")
    replaced = _placeholderize_string(original, first, last)

    # Extract the phone number from the original parsed object (before name replacement).
    parsed_original = json.loads(original)
    phone = _extract_phone(parsed_original)

    # Replace phone value with placeholder in the (already name-replaced) string.
    if phone and phone in replaced:
        replaced = replaced.replace(json.dumps(phone), json.dumps(PHONE_PLACEHOLDER))

    changed = replaced != original

    if changed:
        # Sanity: confirm no name leaks.
        if _word_re(first).search(replaced) or _word_re(last).search(replaced):
            raise RuntimeError(f"{path}: name leaked past placeholderization")
        if not dry_run:
            tmp = path.with_suffix(path.suffix + ".tmp")
            # Re-pretty-print to keep diffs sane.
            tmp.write_text(json.dumps(json.loads(replaced), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    return changed, phone


def migrate_file(path: Path, dry_run: bool) -> tuple[int, int, int]:
    records = []
    changed = 0
    scenario_changed = 0
    domain = path.stem.removesuffix("_dataset")
    scenario_dir = DATA_DIR / f"{domain}_scenarios"

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_changed, first, last = migrate_record(rec)
            changed_scen, phone = migrate_scenario_db(scenario_dir / f"{rec['id']}.json", first, last, dry_run)
            if changed_scen:
                scenario_changed += 1
            # Resolve phone: prefer what was just extracted, fall back to already-stored value.
            phone = phone or rec.get("culture_overrides", {}).get("en", {}).get("phone")
            if phone:
                if "phone" not in rec.get("culture_overrides", {}).get("en", {}):
                    rec.setdefault("culture_overrides", {}).setdefault("en", {})["phone"] = phone
                    rec_changed = True
                # Replace the literal phone value in the record body (e.g. ground_truth.expected_scenario_db).
                # Pop culture_overrides so the stored value isn't itself substituted.
                culture = rec.pop("culture_overrides")
                romanized = rec.pop("romanized_culture_overrides", {})
                rec_str = json.dumps(rec, ensure_ascii=False)
                new_rec_str = rec_str.replace(json.dumps(phone), json.dumps(PHONE_PLACEHOLDER))
                if new_rec_str != rec_str:
                    rec.clear()
                    rec.update(json.loads(new_rec_str))
                    rec_changed = True
                else:
                    rec.clear()
                    rec.update(json.loads(rec_str))
                rec["culture_overrides"] = culture
                rec["romanized_culture_overrides"] = romanized
            if rec_changed:
                changed += 1
            records.append(rec)

    if not dry_run and changed:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)

    return changed, scenario_changed, len(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = args.paths or sorted(DATA_DIR.glob("*_dataset.jsonl"))
    if not targets:
        print("No datasets found", file=sys.stderr)
        return 1

    for path in targets:
        changed, scen_changed, total = migrate_file(path, args.dry_run)
        verb = "would migrate" if args.dry_run else "migrated"
        print(f"{path.name}: {verb} {changed}/{total} records, {scen_changed} scenario DB updates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
