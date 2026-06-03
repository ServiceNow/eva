"""Add a language/culture to an EVA dataset.

Performs all one-time setup needed to run the benchmark in a new language:

  1. **Dataset records** — for each record in ``data/<domain>_dataset.jsonl``:
       - Picks a culturally appropriate (first, last) name pair matching the
         record's gender, deterministically seeded by record id + language.
       - Sets ``record.culture_overrides[<lang>] = {first_name, last_name}``.
       - Translates ``user_goal.en_starting_utterance`` into
         ``user_goal.<lang>_starting_utterance`` via LLM, preserving
         ``<FIRST_NAME>`` / ``<LAST_NAME>`` placeholders.
       - Translates any scenario alias names (facility names, zones, etc.)
         that appear in ``data/<domain>_scenarios/``.

  2. **Agent config** — writes a "respond in <language>" addendum to
     ``configs/agents/language_addenda.yaml`` and the agent's opening greeting
     to ``configs/agents/initial_messages.yaml``.

  3. **WER normalizer** — generates ``wer_normalization/configs/<lang>.json``
     via LLM (number vocabulary, filler words, abbreviations, etc.) and
     optionally a spelling-variation map. New configs are auto-discovered at
     runtime without any further code changes.

  4. **Environment / display** — patches ``.env.example`` with the new
     ``EVA_<LANG>_USER_*`` stubs and registers the display name in
     ``LANGUAGE_DISPLAY_NAMES``.

All steps are idempotent: existing entries are skipped.

Name source (one of):
  --names-file path/to/names.json  containing
      {"male_first": [...], "female_first": [...], "last": [...]}
  --auto-generate-names   ask the LLM for 40+40+40 culturally authentic names
                          (ASCII/romanized + native-script halves).

Key arguments:
  --language            BCP-47 tag (required), e.g. ``fr``, ``es-MX``
  --language-name       Human-readable English name (required), e.g. ``French``
  --native-name         Optional native name shown in the agent addendum, e.g. ``français``
  --domain              Restrict to one domain (repeatable). Default: all domains.
  --auto-generate-names Ask the LLM for 80 culturally authentic names per gender
                        (40 romanized + 40 native-script). Mutually exclusive with
                        --names-file.
  --names-file          Path to a pre-built names JSON:
                            {"male_first": [...], "female_first": [...], "last": [...]}
  --dump-names          Save the LLM-generated name arrays to a file for reuse.
  --include-spelling-variation
                        Also generate a ``{lang}_spelling.json`` that maps regional
                        spelling variants to a canonical form (e.g. ``colour`` →
                        ``color``). Useful for languages with significant dialect
                        orthography divergence. English ships one by default; most
                        other languages don't need it.
  --llm-model           Override the LLM used for generation (default: gpt-5.5).
  --record-id           Mutate only a single record — useful for spot-checking a diff.
  --dry-run             Print what would be written without touching any files.

Usage:
  python scripts/add_culture_data.py --domain airline --language fr \\
      --language-name French --auto-generate-names

  python scripts/add_culture_data.py --domain itsm --language es-MX \\
      --language-name "Mexican Spanish" --names-file es_mx_names.json

  # With regional spelling normalization (e.g. pt-BR vs pt-PT):
  python scripts/add_culture_data.py --domain airline --language pt \\
      --language-name Portuguese --auto-generate-names --include-spelling-variation
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from eva.utils.culture import FIRST_NAME_PLACEHOLDER, LAST_NAME_PLACEHOLDER
from eva.utils.json_utils import extract_and_load_json
from eva.utils.llm_client import LLMClient
from eva.utils.logging import get_logger, setup_logging
from eva.utils.router import init
from eva.utils.wer_normalization.engine import LanguageConfig
from eva.utils.wer_normalization.locale_defaults import locale_defaults
from eva.utils.wer_normalization.wer_utils import normalize_text
from eva.utils.wer_normalization.whisper_normalizer.basic import (
    remove_symbols_and_diacritics,
    remove_symbols_keep_marks,
)

setup_logging()
logger = get_logger(__name__)
load_dotenv()
init(json.loads(os.getenv("EVA_MODEL_LIST")))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ADDENDA_PATH = REPO_ROOT / "configs" / "agents" / "language_addenda.yaml"
INITIAL_MESSAGES_PATH = REPO_ROOT / "configs" / "agents" / "initial_messages.yaml"
WER_CONFIGS_DIR = REPO_ROOT / "src" / "eva" / "utils" / "wer_normalization" / "configs"

DEFAULT_MODEL = "gpt-5.5-2026-04-23"
TRANSLATION_BATCH = 25
ALIAS_BATCH = 50  # Max unique names per alias-translation LLM call.

# Paths within a scenario JSON that may contain name_aliases entries.
_ALIAS_PATHS: list[tuple[str, ...]] = [
    ("facilities", "buildings"),
    ("facilities", "zones"),
    ("software_catalog",),
]

# Template filled at runtime from --language-name (and optionally --native-name).
ADDENDUM_TEMPLATE = (
    "Always respond to the user in {language_name}{native_suffix}, regardless of the instructions given or tool outputs received."
    " However, tool calls and tool names must always be done using ascii characters, except parameters like people's first"
    " or last names which may be in non-ascii, native script. You may need to try both scripts when looking up by name. "
    "All translatable values should be translated when talking to the user. For example, if you are telling the user about "
    "a location from a tool response which says 'downtown', this should be translated. Distinct item names (e.g. 'IntelliJ') "
    "should be kept in their original form."
)


BUCKET_SIZE = 40  # Each name array: indices [0:BUCKET_SIZE] = ASCII, [BUCKET_SIZE:2*BUCKET_SIZE] = native script.


async def _generate_phone_format(language_name: str, language: str, llm: LLMClient) -> dict:
    """Return {calling_code: str, mobile_groups: list[int | str]} for the language's mobile format.

    Each group element is either:
      - int: that many random digits
      - str: a fixed literal prefix (e.g. "6" for French mobiles that always start with 06/+336)

    Example for France: {"calling_code": "33", "mobile_groups": ["6", 2, 2, 2, 2]}
    renders as +33 6 XX XX XX XX.
    Example for UK: {"calling_code": "44", "mobile_groups": ["7", 3, 6]}
    renders as +44 7 XXX XXXXXX.
    """
    prompt = (
        f"For {language_name} ({language}), what is the standard mobile phone number format?\n"
        "Make sure you're aware that the region is the defining factor (eg, pt-BR would use Brazilian phone numbers).\n"
        "Return JSON with:\n"
        "  calling_code: string (country calling code, no leading +)\n"
        "  mobile_format: a format string where X = random digit, fixed digits are written literally,\n"
        "    and separators (spaces, hyphens, dots) are included as-is.\n"
        'Example for France (+33 6 XX XX XX XX): {"calling_code": "33", "mobile_format": "6 XX XX XX XX"}\n'
        'Example for UK (+44 7XXX XXXXXX): {"calling_code": "44", "mobile_format": "7XXX XXXXXX"}\n'
        'Example for US (+1 NXX-NXX-XXXX): {"calling_code": "1", "mobile_format": "XXX-XXX-XXXX"}\n'
        'Example for Germany (+49 151 12345678): {"calling_code": "49", "mobile_format": "151 XXXXXXXX"}\n'
        "Return only the JSON object, no markdown."
    )
    text, _ = await llm.generate_text(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = extract_and_load_json(text)
    calling_code = str(data.get("calling_code", ""))
    mobile_format = data.get("mobile_format")
    if not calling_code or not isinstance(mobile_format, str) or not mobile_format:
        raise ValueError(f"Phone format generation returned invalid data: {data!r}")
    return {"calling_code": calling_code, "mobile_format": mobile_format}


def _render_phone(spec: dict, record_id: str) -> str:
    """Generate a deterministic mobile phone number for record_id using the format spec.

    In mobile_format, each 'X' is replaced by a random digit; all other characters are literals.
    """
    h = int(hashlib.sha256(f"phone:{record_id}".encode()).hexdigest(), 16)
    calling_code = spec["calling_code"]
    result = []
    for ch in spec["mobile_format"]:
        if ch == "X":
            result.append(str(h % 10))
            h //= 10
            if h == 0:
                h = int(hashlib.sha256("".join(result).encode()).hexdigest(), 16)
        else:
            result.append(ch)
    return f"+{calling_code} {''.join(result)}"


def _seeded_index(seed: str, n: int) -> int:
    """Deterministic index in ``[0, n)`` keyed by ``seed``."""
    if n <= 0:
        raise ValueError("Empty name array")
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return h % n


def _use_native_script(record_id: str) -> bool:
    """Fixed, language-independent assignment: ~half of records get native-script names.

    Seeded only on record_id so the same record always picks the same script tier
    regardless of which language is being added.
    """
    h = int(hashlib.sha256(f"script:{record_id}".encode()).hexdigest(), 16)
    return (h % 2) == 1


def _gender_to_bucket(gender: str) -> str:
    g = gender.strip().lower()
    if g in {"man", "male", "m"}:
        return "male_first"
    if g in {"woman", "female", "f"}:
        return "female_first"
    raise ValueError(f"Unexpected gender value {gender!r}")


async def _generate_names(language_name: str, llm: LLMClient) -> dict[str, list[str]]:
    """Two separate requests — one for romanized/ASCII names, one for native-script names.

    Final arrays are [ascii_half] + [native_half] so indices always align with BUCKET_SIZE.
    """

    def _make_prompt(script_instruction: str) -> str:
        return (
            f"Generate culturally authentic {language_name} names for a synthetic dataset.\n"
            "Return JSON with EXACTLY these keys: male_first, female_first, last.\n"
            f"Each list must have EXACTLY {BUCKET_SIZE} names.\n"
            f"Script: {script_instruction}\n"
            "Rules:\n"
            "- No duplicates within a list. No honorifics or titles.\n"
            "- Include common everyday names, not only famous people.\n"
            f'Response format: {{"male_first": [...{BUCKET_SIZE} items...], "female_first": [...{BUCKET_SIZE} items...], "last": [...{BUCKET_SIZE} items...]}}'
        )

    ascii_prompt = _make_prompt("Latin alphabet only, no diacritics (romanized/ASCII forms).")
    native_prompt = _make_prompt(
        "Native script only (e.g. kanji for Japanese, Cyrillic for Russian, Arabic script, Devanagari, etc.). "
        "For languages that only use Latin script (e.g. French, Spanish), use full diacritics."
    )

    (ascii_text, _), (native_text, _) = await asyncio.gather(
        llm.generate_text([{"role": "user", "content": ascii_prompt}], response_format={"type": "json_object"}),
        llm.generate_text([{"role": "user", "content": native_prompt}], response_format={"type": "json_object"}),
    )

    ascii_data = extract_and_load_json(ascii_text)
    native_data = extract_and_load_json(native_text)
    result: dict[str, list[str]] = {}
    for key in ("male_first", "female_first", "last"):
        for label, data in (("ascii", ascii_data), ("native", native_data)):
            lst = data.get(key)
            if not isinstance(lst, list) or len(lst) != BUCKET_SIZE:
                raise ValueError(f"Name generation ({label}): expected {BUCKET_SIZE} items for {key!r}, got {lst!r}")
        result[key] = ascii_data[key] + native_data[key]

    return result


async def _translate_utterances(utterances: list[str], language_name: str, llm: LLMClient) -> list[str]:
    """Batch-translate a list of English utterances. Preserves placeholder tokens."""
    out: list[str] = []
    for i in range(0, len(utterances), TRANSLATION_BATCH):
        chunk = utterances[i : i + TRANSLATION_BATCH]
        numbered = "\n".join(f"{j + 1}. {u}" for j, u in enumerate(chunk))
        prompt = (
            f"Translate each numbered English utterance below into {language_name}.\n"
            "Rules:\n"
            f"- Preserve the literal tokens {FIRST_NAME_PLACEHOLDER} and {LAST_NAME_PLACEHOLDER} verbatim.\n"
            "- Keep numbers, dates, and currency in their original form unless localization is conventional.\n"
            "- Use natural, conversational phrasing as a caller would speak.\n"
            '- Return JSON: {"translations": ["...", "..."]} in the same order.\n\n'
            f"Utterances:\n{numbered}"
        )
        text, _ = await llm.generate_text(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = extract_and_load_json(text)
        translations = data.get("translations")
        if not isinstance(translations, list) or len(translations) != len(chunk):
            raise ValueError(f"Expected {len(chunk)} translations, got: {translations!r}")
        out.extend(translations)
    return out


async def _romanize_names(names: dict[str, list[str]], language_name: str, llm: LLMClient) -> dict[str, list[str]]:
    """Return an ASCII-romanized parallel copy of ``names`` (same shape, same order).

    Only the native-script half (indices BUCKET_SIZE onwards) is sent to the LLM.
    The ASCII half is already correct and is copied through unchanged.
    """
    flat_native: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for key in ("male_first", "female_first", "last"):
        start = len(flat_native)
        flat_native.extend(names[key][BUCKET_SIZE:])
        spans[key] = (start, len(flat_native))

    prompt = (
        f"Romanize each name below into ASCII using standard {language_name} transliteration.\n"
        "Rules:\n"
        "- Preserve order exactly; no additions, no removals, no duplicates collapsed.\n"
        "- Single token per input (no titles, no diacritics in output).\n"
        '- Return JSON: {"romanized": ["...", "..."]}\n\n'
        "Names:\n" + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(flat_native))
    )
    text, _ = await llm.generate_text(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = extract_and_load_json(text)
    rom = data.get("romanized")
    if not isinstance(rom, list) or len(rom) != len(flat_native):
        raise ValueError(f"Expected {len(flat_native)} romanized names, got: {rom!r}")
    # Reconstruct: ASCII half unchanged, native half romanized — indices stay aligned.
    return {key: names[key][:BUCKET_SIZE] + rom[spans[key][0] : spans[key][1]] for key in spans}


def _load_names_file(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("male_first", "female_first", "last"):
        lst = data.get(key)
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"--names-file missing/empty key {key!r}")
        if len(lst) < BUCKET_SIZE * 2:
            raise ValueError(
                f"--names-file key {key!r} has {len(lst)} entries; need at least {BUCKET_SIZE * 2} "
                f"({BUCKET_SIZE} ASCII + {BUCKET_SIZE} native-script)"
            )
    return data


async def _translate_initial_message(language: str, language_name: str, llm: LLMClient) -> str:
    """Translate the English initial message into ``language``."""
    existing: dict[str, str] = {}
    if INITIAL_MESSAGES_PATH.exists():
        existing = yaml.safe_load(INITIAL_MESSAGES_PATH.read_text(encoding="utf-8")) or {}
    if language in existing:
        return existing[language]
    en_message = existing.get("en", "Hello! How can I help you today?")
    prompt = (
        f"Translate the following English greeting into {language_name}.\n"
        "Use natural, conversational phrasing as a voice assistant would speak.\n"
        'Return JSON: {"message": "..."}\n\n'
        f"English: {en_message}"
    )
    text, _ = await llm.generate_text(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = extract_and_load_json(text)
    translated = data.get("message")
    if not translated:
        raise ValueError(f"Initial message translation returned empty result: {data!r}")
    return translated


def _update_initial_messages(language: str, message: str) -> None:
    existing: dict[str, str] = {}
    if INITIAL_MESSAGES_PATH.exists():
        existing = yaml.safe_load(INITIAL_MESSAGES_PATH.read_text(encoding="utf-8")) or {}
    if existing.get(language) == message:
        return
    existing[language] = message
    INITIAL_MESSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    INITIAL_MESSAGES_PATH.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=True), encoding="utf-8")


def _update_addenda(language: str, addendum: str) -> None:
    existing: dict[str, str] = {}
    if ADDENDA_PATH.exists():
        existing = yaml.safe_load(ADDENDA_PATH.read_text(encoding="utf-8")) or {}
    if existing.get(language) == addendum:
        return
    existing[language] = addendum
    ADDENDA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADDENDA_PATH.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=True), encoding="utf-8")


def _get_nested(obj: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for k in keys:
        obj = obj.get(k, {})
    return obj


def _iter_alias_entries(data: dict[str, Any]) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    results = []
    for path in _ALIAS_PATHS:
        section = _get_nested(data, path)
        for entry_key, entry in section.items():
            if entry.get("name_aliases_translatable"):
                results.append((path, entry_key, entry))
    return results


async def _translate_aliases(
    name_to_base: dict[str, list[str]],
    language_name: str,
    llm: LLMClient,
) -> dict[str, list[str]]:
    """Return {canonical_name: [translated_aliases]} for all names in name_to_base.

    Keyed by canonical name so correspondence is guaranteed regardless of batching.
    Translations represent what speakers of language_name would naturally call each item —
    not always a direct translation (e.g. colloquial shorthand may differ).
    """
    items = list(name_to_base.items())
    result: dict[str, list[str]] = {}
    for i in range(0, len(items), ALIAS_BATCH):
        chunk = items[i : i + ALIAS_BATCH]
        payload = dict(chunk)
        prompt = (
            f"You are helping localise a dataset for {language_name} speakers.\n\n"
            "For each entry below, generate natural aliases that a native speaker would use "
            "when referring to that item in a voice call. The input shows the canonical English "
            "name and its English aliases as examples of the kind of shorthand and phrasing to "
            "aim for — produce the equivalent in the target language. These are not limited to direct "
            "translations, but direct translations should be used if appropriate. Use culturally "
            "natural phrasing (colloquial names, common shorthand) where appropriate.\n\n"
            "Rules:\n"
            f"- Generate aliases in {language_name} only; do not repeat the English base aliases.\n"
            "- Each list must have at least 1 alias and at most 4 aliases.\n"
            "- Use lowercase for all aliases if the language has case.\n"
            "- The input keys are canonical English names — preserve them exactly as keys in the output.\n"
            '- Return JSON: {"results": {"<canonical name>": ["alias1", "alias2", ...], ...}}\n\n'
            f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        text, _ = await llm.generate_text(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = extract_and_load_json(text)
        batch_results = data.get("results")
        if not isinstance(batch_results, dict):
            raise ValueError(f"Expected dict under 'results', got: {batch_results!r}")
        for name, _ in chunk:
            if name not in batch_results:
                raise ValueError(f"LLM did not return aliases for name {name!r}")
            aliases = batch_results[name]
            if not isinstance(aliases, list) or not aliases:
                raise ValueError(f"Empty or invalid aliases for {name!r}: {aliases!r}")
            result[name] = [a.lower().strip() for a in aliases]
    return result


async def add_scenario_aliases(
    domain: str,
    language: str,
    language_name: str,
    llm: LLMClient,
    dry_run: bool,
) -> None:
    """Translate name_aliases for tagged translatable entries in scenario DB JSONs.

    Idempotent: aliases already present in name_aliases are not re-added.
    Requires migrate_aliases.py to have been run first (entries must have
    name_aliases_translatable and name_aliases_base).
    """
    scenario_dir = DATA_DIR / f"{domain}_scenarios"
    if not scenario_dir.exists():
        logger.info(f"No scenario directory for domain={domain!r}, skipping alias translation")
        return

    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        return

    # Collect unique translatable names and their base aliases across all files.
    name_to_base: dict[str, list[str]] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for _, _, entry in _iter_alias_entries(data):
            name = entry["name"]
            if name not in name_to_base:
                name_to_base[name] = entry.get("name_aliases_base", entry.get("name_aliases", []))

    if not name_to_base:
        logger.info(f"No translatable alias entries found for domain={domain!r}")
        return

    logger.info(f"Translating aliases for {len(name_to_base)} unique names to {language_name}")
    translated = await _translate_aliases(name_to_base, language_name, llm)

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = False

        for _, _, entry in _iter_alias_entries(data):
            new_aliases = translated.get(entry["name"], [])
            existing = set(entry.get("name_aliases", []))
            to_add = [a for a in new_aliases if a not in existing]
            if to_add:
                entry["name_aliases"].extend(to_add)
                entry["name_aliases"].sort()
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
    dataset_path = DATA_DIR / f"{domain}_dataset.jsonl"
    if dataset_path.exists():
        records: list[dict[str, Any]] = []
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
                new_aliases = translated.get(entry["name"], [])
                existing = set(entry.get("name_aliases", []))
                to_add = [a for a in new_aliases if a not in existing]
                if to_add:
                    entry["name_aliases"].extend(to_add)
                    entry["name_aliases"].sort()
                    dataset_changed = True

        if dataset_changed:
            if dry_run:
                logger.info(f"[dry-run] would update {dataset_path.name} with translated aliases")
            else:
                tmp = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.replace(dataset_path)
                logger.info(f"Updated {dataset_path.name} with translated aliases")


async def add_culture(
    domain: str,
    language: str,
    language_name: str,
    names: dict[str, list[str]] | None,
    romanized_names: dict[str, list[str]] | None,
    llm: LLMClient,
    dry_run: bool,
    addendum: str,
    record_id: str | None = None,
    phone_spec: dict | None = None,
) -> None:
    dataset_path = DATA_DIR / f"{domain}_dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if record_id is not None:
        matching = [r for r in records if r.get("id") == record_id]
        if not matching:
            raise ValueError(f"No record with id={record_id!r} in {dataset_path}")
        target_ids = {record_id}
        logger.info(f"Scoped to record_id={record_id} ({domain})")
    else:
        target_ids = {r["id"] for r in records}

    # 1. Assign names per record (deterministic).
    to_translate_idx: list[int] = []
    to_translate_text: list[str] = []
    for idx, rec in enumerate(records):
        if rec.get("id") not in target_ids:
            continue
        rec.setdefault("culture_overrides", {})
        # Must have 'en' from Phase A migration.
        if "en" not in rec["culture_overrides"]:
            raise ValueError(
                f"Record {rec.get('id')!r} missing culture_overrides.en — run migrate_to_culture_schema.py first"
            )
        gender = (rec.get("user_config") or {}).get("gender")
        if not gender:
            raise ValueError(f"Record {rec.get('id')!r} missing user_config.gender")
        bucket = _gender_to_bucket(gender)

        if language not in rec["culture_overrides"]:
            if names is None or romanized_names is None:
                raise ValueError(
                    f"Record {rec.get('id')!r} missing culture_overrides[{language!r}] but no name arrays available — "
                    "re-run without skipping name generation"
                )
            seed = f"{rec['id']}|{language}"
            # Script tier is fixed per record_id across all languages (~50/50 ASCII vs native).
            offset = BUCKET_SIZE if _use_native_script(rec["id"]) else 0
            first_idx = offset + _seeded_index(seed + "|first", BUCKET_SIZE)
            last_idx = offset + _seeded_index(seed + "|last", BUCKET_SIZE)
            rec["culture_overrides"][language] = {
                "first_name": names[bucket][first_idx],
                "last_name": names["last"][last_idx],
            }
            rec.setdefault("romanized_culture_overrides", {})[language] = {
                "first_name": romanized_names[bucket][first_idx],
                "last_name": romanized_names["last"][last_idx],
            }

        # Assign phone number for airline domain (deterministic per record).
        if phone_spec and "phone" not in rec["culture_overrides"].get(language, {}):
            rec["culture_overrides"].setdefault(language, {})["phone"] = _render_phone(phone_spec, rec["id"])

        rec.setdefault("starting_utterances", {})
        if "en" not in rec["starting_utterances"]:
            raise ValueError(
                f"Record {rec.get('id')!r} missing starting_utterances.en — run migrate_to_culture_schema.py first"
            )
        if language not in rec["starting_utterances"]:
            to_translate_idx.append(idx)
            to_translate_text.append(rec["starting_utterances"]["en"])

    # 2. Translate utterances in batch.
    if to_translate_text:
        logger.info(f"Translating {len(to_translate_text)} utterances to {language_name}")
        translated = await _translate_utterances(to_translate_text, language_name, llm)
        for idx, t in zip(to_translate_idx, translated, strict=True):
            records[idx]["starting_utterances"][language] = t

    # 3. Write back atomically.
    if dry_run:
        logger.info(f"[dry-run] would update {dataset_path} ({len(target_ids)} records)")
    else:
        tmp = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(dataset_path)
        logger.info(f"Updated {dataset_path}")

    # 4. Agent addendum (template filled from CLI args).
    logger.info(f"Addendum for {language}: {addendum}")
    if not dry_run:
        _update_addenda(language, addendum)
        logger.info(f"Updated {ADDENDA_PATH}")


def _all_records_have_language(language: str, domains: list[str], record_id: str | None) -> bool:
    """Return True if every target record already has culture_overrides[language] with first/last name."""
    for domain in domains:
        path = DATA_DIR / f"{domain}_dataset.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if record_id and rec.get("id") != record_id:
                    continue
                entry = rec.get("culture_overrides", {}).get(language, {})
                if not entry.get("first_name") or not entry.get("last_name"):
                    return False
    return True


def _all_airline_records_have_phone(language: str, record_id: str | None) -> bool:
    """Return True if every target airline record already has culture_overrides[language].phone."""
    path = DATA_DIR / "airline_dataset.jsonl"
    if not path.exists():
        return True
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if record_id and rec.get("id") != record_id:
                continue
            if not rec.get("culture_overrides", {}).get(language, {}).get("phone"):
                return False
    return True


async def amain(args: argparse.Namespace) -> int:
    llm = LLMClient(model=args.llm_model, params={"temperature": 0.0})

    domains = args.domains or [p.stem.removesuffix("_dataset") for p in sorted(DATA_DIR.glob("*_dataset.jsonl"))]

    if args.auto_generate_names:
        if _all_records_have_language(args.language, domains, args.record_id):
            logger.info(f"All records already have {args.language} names — skipping name generation")
            names = None
            romanized_names = None
        else:
            logger.info(f"Generating {args.language_name} name arrays via LLM")
            names = await _generate_names(args.language_name, llm)
            if args.dump_names:
                Path(args.dump_names).write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"Wrote generated names to {args.dump_names}")
            logger.info("Romanizing name arrays via LLM")
            romanized_names = await _romanize_names(names, args.language_name, llm)
            if args.dump_names:
                out = Path(args.dump_names)
                rom_out = out.with_name(out.stem + "_romanized" + out.suffix)
                rom_out.write_text(json.dumps(romanized_names, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"Wrote romanized names to {rom_out}")
    else:
        names = _load_names_file(Path(args.names_file))
        logger.info("Romanizing name arrays via LLM")
        romanized_names = await _romanize_names(names, args.language_name, llm)
        if args.dump_names:
            out = Path(args.dump_names)
            rom_out = out.with_name(out.stem + "_romanized" + out.suffix)
            rom_out.write_text(json.dumps(romanized_names, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Wrote romanized names to {rom_out}")

    native_suffix = f" ({args.native_name})" if args.native_name else ""
    addendum = ADDENDUM_TEMPLATE.format(language_name=args.language_name, native_suffix=native_suffix)

    logger.info(f"Translating initial message to {args.language_name}")
    initial_message = await _translate_initial_message(args.language, args.language_name, llm)
    logger.info(f"Initial message for {args.language}: {initial_message}")
    if not args.dry_run:
        _update_initial_messages(args.language, initial_message)
        logger.info(f"Updated {INITIAL_MESSAGES_PATH}")

    for domain in domains:
        logger.info(f"=== Domain: {domain} ===")
        if domain == "airline":
            if _all_airline_records_have_phone(args.language, args.record_id):
                logger.info(
                    f"All airline records already have {args.language} phone — skipping phone format generation"
                )
                phone_spec = None
            else:
                logger.info(f"Generating phone format spec for {args.language_name}")
                phone_spec: dict | None = await _generate_phone_format(args.language_name, args.language, llm)
                logger.info(f"Phone format for {args.language_name}: {phone_spec}")
        else:
            phone_spec = None
        await add_culture(
            domain,
            args.language,
            args.language_name,
            names,
            romanized_names,
            llm,
            args.dry_run,
            addendum,
            args.record_id,
            phone_spec,
        )
        await add_scenario_aliases(domain, args.language, args.language_name, llm, args.dry_run)

    update_env_example(args.language, args.language_name, REPO_ROOT / ".env.example", args.dry_run)
    update_language_display_names(
        args.language, args.language_name, REPO_ROOT / "src" / "eva" / "models" / "config.py", args.dry_run
    )
    await update_wer_normalizer_config(
        args.language,
        args.language_name,
        WER_CONFIGS_DIR,
        llm,
        args.dry_run,
        args.include_spelling_variation,
    )
    return 0


def _normalize_lang(language: str) -> str:
    """Normalise to pipecat's BCP 47 casing: lowercase base, uppercase region.

    'fr-ca' -> 'fr-CA', 'FR-CA' -> 'fr-CA', 'fr' -> 'fr'
    """
    parts = re.split(r"[-_]", language, maxsplit=1)
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return parts[0].lower()


def _lang_to_env_prefix(language: str) -> str:
    """Convert BCP 47 tag to a valid env-var prefix segment.

    'fr' -> 'FR', 'fr-CA' -> 'FR_CA'
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", language).upper().strip("_")


_WER_PROMPT = """You are configuring a number-word WER normalizer for {language_name} (BCP-47: {language}).

The normalizer converts spelled-out numbers to digits for ASR/STT evaluation
("twenty two" → "22"). It is a left-to-right state machine driven by vocabulary
tables, with one optional preprocessor that handles units-before-tens languages.

=== STEP 1: classify the number system ===

Choose exactly one "family":
  - "alphabetic_ltr": compositional left-to-right base-10. Tens word comes first,
    then ones (English "twenty two", Spanish "veintidós", Hindi compositional ≥100).
    French-style vigesimal (70=60+10, 80=4*20) is also this family — set vigesimal.
  - "alphabetic_reversed_units": units come before tens, optionally glued by a
    conjunction word (German "einundzwanzig", Dutch "eenentwintig", Arabic
    "wahid wa-'ishrun", Hebrew "echad ve-esrim").
  - "lexicalized_below_100": numbers 1-99 are mostly distinct words rather than
    composed (Hindi/Bengali native form). Put every word 1-99 in "cardinals",
    leave "tens" empty. Engine still handles 100+ compositionally.
  - "cjk": positional with 万/億 grouping (ZH/JA/KO). NOT supported by this script —
    return {{"family": "cjk", "reason": "..."}} and stop.
  - "unsupported": any other system. Return {{"family": "unsupported", "reason": "..."}}.

=== STEP 2: vocabulary (the bulk of your work) ===

The engine has two kinds of number-word entries:

  (A) FIXED-VALUE words — a single word that means a single specific integer.
      The engine looks them up directly. "twelve"=12, "thirty"=30. If a language
      has a dedicated single word for, say, 500 (one token, not "five hundred"),
      that's also a fixed-value word with value 500.

  (B) SCALING words — words that multiply the running total by some factor.
      "hundred", "thousand", "million", "lakh", "crore". These are NEVER
      standalone values; they always scale a preceding cardinal ("two hundred"
      = 2 × 100). If the word never appears alone meaning that exact number,
      it's a scaling word.

  ⚠️ The single most common error is putting a fixed-value compound (e.g. a
  single-token word for 500) into the scaling list because "it equals 500".
  Test: can the word stand alone meaning exactly that value? If yes → fixed.
  Can it be preceded by a cardinal to scale it ("five [X]")? If yes → scaling.

⚠️ A word must appear in EXACTLY ONE of {{cardinals, tens, scaling_units}}.
The engine checks them in that order; a word in both `cardinals` and `tens`
will be treated as a cardinal and break composition.

Return these dicts under "vocabulary":
  - "zeros": list of words meaning zero (e.g. ["zero", "oh"]).
  - "cardinals": {{word: int}} fixed-value cardinal words that are NOT a
    multiple of 10. Specifically:
      • alphabetic_ltr / alphabetic_reversed_units: 1-19 (English: one…nineteen,
        French: un…seize plus "dix-sept" etc. if the language uses them as one
        token). Plus any lexicalized larger values that aren't multiples of 10
        — e.g. lexicalized hundreds 200, 300, 500, 700 … when expressed as
        single words.
      • lexicalized_below_100: every distinct word 1-99 (no `tens` needed).
    NEVER put 20/30/…/90 in cardinals — they go in `tens`.
  - "cardinal_variants": {{word: int}} alternate surface forms of the SAME
    integer (gender/case/plural — French "une"=1, German "ein"=1, regional
    Swiss/Belgian forms like French "septante"=70, "huitante"=80, "nonante"=90).
  - "cardinals_suffixed": {{word: [int, suffix]}} ordinal/plural surface forms
    that should emit a suffixed digit. STRONGLY ENCOURAGED for languages with
    a productive ordinal pattern (most European languages do — English, French,
    German, Spanish, Italian, Polish …). Examples:
      English: {{"first": [1, "st"], "second": [2, "nd"], "third": [3, "rd"],
                "fourth": [4, "th"], …}}
      French:  {{"premier": [1, "er"], "première": [1, "ère"],
                 "deuxième": [2, "ème"], "troisième": [3, "ème"], …}}
      German:  {{"erste": [1, "."], "zweite": [2, "."], "dritte": [3, "."], …}}
    Cover at least 1-19 + key ordinals like "twentieth". Leave {{}} only for
    languages without a regular ordinal-suffix construction.
  - "tens": {{word: int}} ALL multiples of 10 from 20 to 90 (or as many as the
    language has — French stops at 60 because 70-90 are vigesimal). Include
    plural/inflected forms here too: French "vingt"=20 AND "vingts"=20.
    Empty {{}} for lexicalized_below_100.
  - "tens_suffixed": same shape as cardinals_suffixed, for tens. Optional.
  - "scaling_units": {{word: int}} multiplicative scaling words ONLY: "hundred",
    "thousand", "million", "billion", "lakh", "crore". A word belongs here
    only if it scales a preceding number rather than naming a fixed value.
  - "scaling_units_suffixed": optional, same shape.

=== STEP 3: connective words ===

  - "conjunction_word": the spoken word that joins large+small in this language
    ("and"/"et"/"und"/"wa"/"y"). null if numbers never use one.
  - "decimal_word": the spoken decimal separator ("point"/"virgule"/"komma"). null if none.

=== STEP 4: structural flags ===

  - "split_hyphenated_numbers": true if the language writes numbers like
    "quatre-vingt-dix" that must be split into tokens. False otherwise.
  - "vigesimal": only for languages with vigesimal compositions like
    French/Belgian-French/Swiss-French (where 70=60+10, 80=4*20, 90=4*20+10).
    null for all other languages — including those that just have a vigesimal
    word floating around without compositional use.
    Format (each field is necessary, do not omit any):
      {{
        "trigger_words": [...],        // tens-words that participate in the
                                        // multiplication. Include ALL surface
                                        // forms. French: ["vingt", "vingts"].
        "residuals": [...],            // values mod-100 that multiply the
                                        // trigger word. French: [4,5,6,7,8,9]
                                        // (because quatre/cinq/.../neuf × vingt
                                        // = 80, 100 (handled elsewhere), …, 180).
        "additive_teen_residuals": [...] // values mod-100 at which a following
                                          // teen ADDS rather than concatenates.
                                          // French: [60, 80] — soixante (60) +
                                          // dix (10) = 70, quatre-vingt (80) +
                                          // dix (10) = 90. (0 is added by the
                                          // engine automatically; you can omit it.)
      }}
    Worked French example:
      "quatre-vingt-dix-neuf" → tokens ["quatre","vingt","dix","neuf"]
        quatre → value=4
        vingt  → trigger fires (4 in residuals): value = 0 + 4*20 = 80
        dix    → 80 in additive_teen_residuals: value = 80 + 10 = 90
        neuf   → value % 10 == 0: value = 90 + 9 = 99  ✓

=== OUTPUT FORMAT ===

Return one JSON object with keys: family, vocabulary, conjunction_word,
decimal_word, split_hyphenated_numbers, vigesimal. Optionally a
"reason" field. No markdown, no commentary.

Reminder on vocabulary shape:
{{
  "zeros": [...],
  "cardinals": {{...}}, "cardinal_variants": {{...}}, "cardinals_suffixed": {{...}},
  "tens": {{...}}, "tens_suffixed": {{...}},
  "scaling_units": {{...}}, "scaling_units_suffixed": {{...}}
}}"""


_WER_TEST_CASES_PROMPT = """Generate 15 number round-trip test cases for {language_name} (BCP-47: {language}).

Each test case is a [spelled_form, digit_form] pair where spelled_form is how a speaker
of {language_name} would say the number out loud, and digit_form is the numeric string it
represents.

Cover: small numbers (1-9), teens, tens, a compound like 21 or 22, 100, 121, 1000, 1500,
a decimal number, and at least one language-specific quirk (e.g. vigesimal forms in French,
reversed units in German, long scale in some European languages).

Return JSON: {{"test_cases": [["spelled_form", "digit_form"], ...]}}
No markdown, no commentary."""


_WER_RETRY_PROMPT = """The config you generated has {fail_count} failing round-trip test(s):

{failures}

Each line shows: spelled-out form → what the normalizer produced | digit form → what it produced.
They should match but don't.

Return the COMPLETE corrected JSON object (same schema as before — all keys, full vocabulary tables).
No markdown, no commentary."""


_TEXT_RULES_PROMPT = """For {language_name} (BCP-47: {language}), provide two short
lists used to clean STT transcripts before WER comparison. These are plain
data — do NOT write regex syntax; we wrap them server-side.

=== filler_words ===
A list of non-lexical hesitation/disfluency words that should be removed
before comparison. English equivalents: um, uh, hmm, mhm, er, ah.
Include only true fillers — NOT real content words that happen to be short.
Cap at ~12 entries.

Each entry: a single token in the native script of the language. Any
script is fine (Latin, Cyrillic, Devanagari, Arabic, Hebrew, Hangul,
CJK, etc.). No spaces, no digits, no punctuation. Diacritics are OK
(they'll be normalized upstream, you don't need to strip them).

Examples:
  French:   ["euh", "ben", "bah", "hein", "hum"]
  German:   ["äh", "ähm", "hm", "naja"]
  Spanish:  ["eh", "este", "bueno", "pues"]
  Russian:  ["ну", "э", "эм", "вот"]
  Hindi:    ["अरे", "हाँ", "वो", "मतलब"]
  Arabic:   ["يعني", "اه", "ام"]
  Japanese: ["えーと", "あの", "まあ"]

=== abbreviations ===
Map of {{abbreviation: spelled_out_form}} for titles/honorifics likely to
appear in STT output in either form. The normalizer rewrites the
abbreviation to its expansion so both forms collapse during WER scoring.

Constraints:
  - both sides: single tokens in the native script, no spaces, no dots,
    no digits, no punctuation
  - cap at ~15 entries
  - only include titles/honorifics genuinely common in this language

Examples:
  English: {{"mr": "mister", "mrs": "missus", "dr": "doctor", "prof": "professor"}}
  French:  {{"m": "monsieur", "mme": "madame", "mlle": "mademoiselle", "dr": "docteur"}}
  German:  {{"hr": "herr", "fr": "frau", "dr": "doktor"}}
  Spanish: {{"sr": "señor", "sra": "señora", "dr": "doctor"}}
  Hindi:   {{"डॉ": "डॉक्टर", "श्री": "श्रीमान"}}

If the language doesn't have widely-abbreviated honorifics, return an empty dict.

=== OUTPUT ===
Return JSON: {{"filler_words": [...], "abbreviations": {{...}}}}.
No markdown, no commentary."""


def _is_letters_only(s: str) -> bool:
    """True iff ``s`` is non-empty and every character is a Unicode letter.

    Covers Latin, Cyrillic, Greek, Arabic, Hebrew, Devanagari, Bengali,
    Tamil, Thai, Hangul, CJK ideographs, etc. Excludes digits, whitespace,
    punctuation, symbols, and combining marks (the runtime stripper removes
    combining marks anyway, so post-stripped tokens are pure letters).
    """
    if not s:
        return False
    return all(unicodedata.category(c).startswith("L") for c in s)


def _normalize_rule_token(raw: str, preserve_marks: bool) -> str | None:
    """Lowercase + script-aware strip a candidate token, then validate.

    Mirrors the upstream text pipeline (which uses ``remove_symbols_keep_marks``
    for Indic/Arabic/Hebrew/Thai and ``remove_symbols_and_diacritics`` for
    everything else) so vocab entries match runtime text.
    """
    if not isinstance(raw, str):
        return None
    strip_fn = remove_symbols_keep_marks if preserve_marks else remove_symbols_and_diacritics
    cleaned = strip_fn(raw.strip().lower(), keep="")
    return cleaned if _is_letters_only(cleaned) else None


def _build_text_rules(rules: dict, preserve_marks: bool) -> tuple[str, dict[str, str]]:
    """Convert LLM data → (ignore_patterns, replacers) with regex compile-check.

    Filler words become a single ``\\b(w1|w2|...)\\b`` alternation. Abbreviations
    become individual ``\\b<abbr>\\b`` → expansion replacers. Entries that are
    not pure Unicode letters after lowercase + diacritic strip are dropped.
    """
    fillers: list[str] = []
    for w in rules.get("filler_words") or []:
        tok = _normalize_rule_token(w, preserve_marks)
        if tok:
            fillers.append(tok)

    ignore_pattern = ""
    if fillers:
        # Deduplicate while preserving order
        fillers = list(dict.fromkeys(fillers))
        ignore_pattern = rf"\b({'|'.join(re.escape(w) for w in fillers)})\b"

    replacers: dict[str, str] = {}
    for abbr, expansion in (rules.get("abbreviations") or {}).items():
        a = _normalize_rule_token(abbr, preserve_marks)
        e = _normalize_rule_token(expansion, preserve_marks)
        if not a or not e:
            continue
        replacers[rf"\b{re.escape(a)}\b"] = e

    if ignore_pattern:
        try:
            re.compile(ignore_pattern)
        except re.error as exc:
            logger.warning(f"Dropping invalid ignore_patterns regex: {exc}")
            ignore_pattern = ""
    for pat in list(replacers):
        try:
            re.compile(pat)
        except re.error as exc:
            logger.warning(f"Dropping invalid replacer regex {pat!r}: {exc}")
            replacers.pop(pat, None)

    return ignore_pattern, replacers


def _build_full_config(language: str, llm_data: dict) -> dict:
    """Build a complete LanguageConfig-shaped dict from LLM output + defaults.

    Merges LLM-creative output, locale defaults, and deterministic defaults.

    The LLM is asked for the small creative surface; everything else is
    injected here so we never depend on LLM correctness for structural fields.
    """
    family = llm_data.get("family")
    vocab = llm_data.get("vocabulary") or {}
    vig = llm_data.get("vigesimal") or {}

    # Safety net: the engine matches cardinals before tens, so a word in both
    # gets misclassified (commonly when the LLM puts "vingt" or other multiples
    # of 10 in cardinals). Strip any cardinal that's also in tens with the same
    # value — this is a deterministic correction, not a guess.
    cardinals = dict(vocab.get("cardinals", {}))
    tens = dict(vocab.get("tens", {}))
    for word in list(cardinals):
        if word in tens and cardinals[word] == tens[word]:
            cardinals.pop(word)

    # 0 is always an additive residual (a clean hundred-multiple should add).
    additive = sorted({0, *vig.get("additive_teen_residuals", [])})

    cfg: dict = {
        "code": language,
        "zeros": vocab.get("zeros", []),
        "cardinals": cardinals,
        "cardinal_variants": vocab.get("cardinal_variants", {}),
        "cardinals_suffixed": vocab.get("cardinals_suffixed", {}),
        "tens": tens,
        "tens_suffixed": vocab.get("tens_suffixed", {}),
        "scaling_units": vocab.get("scaling_units", {}),
        "scaling_units_suffixed": vocab.get("scaling_units_suffixed", {}),
        # Connectives
        "conjunction_word": llm_data.get("conjunction_word"),
        "decimal_word": llm_data.get("decimal_word"),
        # Structural flags
        "reversed_units": family == "alphabetic_reversed_units",
        "split_hyphenated_numbers": bool(llm_data.get("split_hyphenated_numbers")),
        # Vigesimal — only set when the LLM explicitly classifies the language.
        "vigesimal_trigger_words": vig.get("trigger_words", []),
        "vigesimal_multiplier": 20,
        "vigesimal_residuals": vig.get("residuals", []),
        "additive_teen_residuals": additive,
        "cardinal_continuation_on_prev_cardinal": False,
        "conjunction_ignore_prev": ["scaling_units", "tens"] if llm_data.get("conjunction_word") else [],
        "repeat_words": {},
        "half_pattern": None,
        "half_replacement": None,
        "one_word": None,
        "one_plural_suffix": "",
        "cents_connector": None,
        # Currency/sign — left empty; user curates if needed
        "preceding_prefixers": {},
        "following_prefixers": {},
        "suffixers": {},
        # Outer text normalizer — populated by a separate LLM call (see
        # _generate_wer_config). Stay empty until that call merges them in.
        "ignore_patterns": "",
        "replacers": {},
        "strip_space_before_apostrophe": False,
        "ordinal_suffix_pattern": "",
        "spelling_map_path": None,
    }
    cfg.update(locale_defaults(language))
    return cfg


async def _generate_wer_config(
    language: str,
    language_name: str,
    configs_dir: Path,
    llm: LLMClient,
    dry_run: bool,
) -> None:
    """Ask the LLM for the linguistic part of a WER config; inject everything else.

    Pipeline:
      1. Single LLM call returns {family, vocabulary, connectives, flags, test_cases}.
      2. If family is "cjk" or "unsupported", abort with a clear message.
      3. Merge with deterministic defaults + BCP-47 locale defaults.
      4. Validate against LanguageConfig.
      5. Round-trip every test case through the resulting normalizer; report
         pass rate. Write the config regardless so the user has a starting point.
    """
    prompt = _WER_PROMPT.format(language_name=language_name, language=language)
    test_cases_prompt = _WER_TEST_CASES_PROMPT.format(language_name=language_name, language=language)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    (text, _), (test_cases_text, _) = await asyncio.gather(
        llm.generate_text(messages, response_format={"type": "json_object"}),
        llm.generate_text([{"role": "user", "content": test_cases_prompt}], response_format={"type": "json_object"}),
    )
    llm_data = extract_and_load_json(text)
    test_cases_data = extract_and_load_json(test_cases_text)
    llm_data["test_cases"] = test_cases_data.get("test_cases") or []

    family = llm_data.get("family")
    if family in {"cjk", "unsupported"}:
        reason = llm_data.get("reason", "no reason given")
        raise ValueError(
            f"LLM classified {language_name!r} as {family!r} — this script only "
            f"supports alphabetic families. Reason: {reason}. "
            f"For CJK languages, add a dedicated normalizer class in cjk.py."
        )
    if family not in {"alphabetic_ltr", "alphabetic_reversed_units", "lexicalized_below_100"}:
        raise ValueError(f"LLM returned unexpected family {family!r}")

    cfg = _build_full_config(language, llm_data)

    rules_prompt = _TEXT_RULES_PROMPT.format(language_name=language_name, language=language)
    try:
        rules_text, _ = await llm.generate_text(
            [{"role": "user", "content": rules_prompt}],
            response_format={"type": "json_object"},
        )
        rules_data = extract_and_load_json(rules_text)
        ignore_pattern, replacers = _build_text_rules(
            rules_data, preserve_marks=bool(cfg.get("preserve_combining_marks"))
        )
        cfg["ignore_patterns"] = ignore_pattern
        cfg["replacers"] = replacers
        logger.info(
            f"Text rules for {language}: {len(replacers)} abbreviations, "
            f"{'filler regex set' if ignore_pattern else 'no fillers'}"
        )
    except Exception as exc:
        logger.warning(
            f"Text-rules generation for {language} failed ({exc}); "
            f"config will have empty ignore_patterns and replacers."
        )

    try:
        validated = LanguageConfig.model_validate(cfg)
    except ValidationError as exc:
        raise ValueError(f"Generated WER config for {language!r} failed schema validation:\n{exc}") from exc
    if validated.code != language:
        raise ValueError(f"Internal: code mismatch {validated.code!r} vs {language!r}")

    out = configs_dir / f"{language}.json"
    if dry_run:
        logger.info(f"[dry-run] would write WER config to {out}")
        return

    configs_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"Wrote WER normalizer config to {out}")

    test_cases = llm_data.get("test_cases") or []
    pass_count, failures = _run_wer_round_trip_tests(language, test_cases)
    total = len(test_cases)
    if total:
        logger.info(f"Round-trip tests for {language}: {pass_count}/{total} passed")
        for spelled, digits, spelled_norm, digits_norm in failures:
            logger.warning(f"  FAIL: {spelled!r} -> {spelled_norm!r}  vs  {digits!r} -> {digits_norm!r}")

    if failures and not dry_run:
        logger.info(f"Retrying with {len(failures)} failure(s) shown to the model …")
        failure_lines = "\n".join(
            f"  {spelled!r} -> {sn!r}  |  {digits!r} -> {dn!r}" for spelled, digits, sn, dn in failures
        )
        retry_prompt = _WER_RETRY_PROMPT.format(fail_count=len(failures), failures=failure_lines)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": retry_prompt})
        retry_text, _ = await llm.generate_text(messages, response_format={"type": "json_object"})
        retry_data = extract_and_load_json(retry_text)

        retry_cfg = _build_full_config(language, retry_data)
        retry_cfg["ignore_patterns"] = cfg.get("ignore_patterns")
        retry_cfg["replacers"] = cfg.get("replacers", {})
        try:
            LanguageConfig.model_validate(retry_cfg)
        except ValidationError as exc:
            logger.warning(f"Retry config failed validation ({exc}); keeping original.")
        else:
            out.write_text(json.dumps(retry_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            logger.info(f"Overwrote {out.name} with retry config")
            retry_pass, retry_failures = _run_wer_round_trip_tests(language, test_cases)
            logger.info(f"Round-trip tests after retry: {retry_pass}/{total} passed")
            for spelled, digits, spelled_norm, digits_norm in retry_failures:
                logger.warning(f"  FAIL: {spelled!r} -> {spelled_norm!r}  vs  {digits!r} -> {digits_norm!r}")
            if retry_failures:
                logger.warning(f"Inspect {language}.json and fix vocabulary entries for failing cases above.")
    elif pass_count < total:
        logger.warning(f"Inspect {language}.json and fix vocabulary entries for failing cases above.")


def _run_wer_round_trip_tests(language: str, test_cases: list) -> tuple[int, list[tuple[str, str, str, str]]]:
    """Verify normalize(spelled) == normalize(digits) under the full pipeline."""
    passed = 0
    failures: list[tuple[str, str, str, str]] = []
    for pair in test_cases:
        if not (isinstance(pair, list) and len(pair) == 2):
            continue
        spelled, digits = str(pair[0]), str(pair[1])
        try:
            sn = normalize_text(spelled, language)
            dn = normalize_text(digits, language)
        except Exception as exc:
            failures.append((spelled, digits, f"<error: {exc}>", ""))
            continue
        if sn.strip() == dn.strip():
            passed += 1
        else:
            failures.append((spelled, digits, sn, dn))
    return passed, failures


async def _generate_spelling_map(
    language: str,
    language_name: str,
    configs_dir: Path,
    llm: LLMClient,
    dry_run: bool,
) -> None:
    """Generate a spelling-variation map ``{variant: canonical}`` for ``language``.

    This is analogous to ``en_spelling.json`` (British→American equivalences).
    Only warranted for languages with significant regional spelling divergence.
    """
    prompt = f"""You are generating a spelling-variation normalization map for {language_name} (BCP-47: {language}).

This map is used during WER evaluation to collapse regional spelling variants into a
single canonical form, so that both sides of a word-error-rate comparison normalize
to the same string (e.g. English "colour" → "color").

=== TASK ===
Return a JSON object where each key is a regional/variant spelling and each value is
the preferred canonical form to normalize to.

Only include pairs where:
- Both forms are correct spellings of the same word in real usage.
- The variant and canonical forms are genuinely different strings.
- The variant is likely to appear in STT output or reference transcripts.

Aim for completeness: include all systematic spelling divergences you know of for
{language_name} (e.g. orthographic reform variants, regional differences between
major dialect regions, common alternative spellings recognized by major dictionaries).

=== FORMAT ===
Return ONLY a flat JSON object: {{"variant_spelling": "canonical_spelling", ...}}
No markdown, no nesting, no surrounding text.

If {language_name} has no significant spelling variants (i.e. spelling is standardized
with fewer than ~10 meaningful pairs), return an empty object: {{}}"""

    text, _ = await llm.generate_text(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = extract_and_load_json(text)
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"Spelling map for {language!r} is not a flat {{str: str}} dict: {type(data)}")

    out = configs_dir / f"{language}_spelling.json"
    if dry_run:
        logger.info(f"[dry-run] would write spelling map ({len(data)} entries) to {out}")
        return

    configs_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"Wrote spelling map ({len(data)} entries) to {out}")


async def update_wer_normalizer_config(
    language: str,
    language_name: str,
    configs_dir: Path,
    llm: LLMClient,
    dry_run: bool,
    include_spelling_variation: bool,
) -> None:
    """Generate the WER normalizer JSON config for *language*.

    Steps:
    1. Generate ``configs/{language}.json`` via LLM (validated against LanguageConfig schema).
    2. Optionally generate ``configs/{language}_spelling.json`` when --include-spelling-variation.
    """
    config_path = configs_dir / f"{language}.json"
    if config_path.exists():
        logger.info(f"WER config already exists at {config_path} — skipping generation")
    else:
        logger.info(f"Generating WER normalizer config for {language_name}")
        await _generate_wer_config(language, language_name, configs_dir, llm, dry_run)

    if include_spelling_variation:
        spelling_path = configs_dir / f"{language}_spelling.json"
        if spelling_path.exists():
            logger.info(f"Spelling map already exists at {spelling_path} — skipping")
        else:
            logger.info(f"Generating spelling-variation map for {language_name}")
            await _generate_spelling_map(language, language_name, configs_dir, llm, dry_run)
            # Point the config's spelling_map_path to the new file (patch in-place).
            if not dry_run and config_path.exists():
                cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
                cfg_data["spelling_map_path"] = f"{language}_spelling.json"
                config_path.write_text(json.dumps(cfg_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                logger.info(f"Patched spelling_map_path in {config_path.name}")


def update_language_display_names(language: str, language_name: str, config_path: Path, dry_run: bool) -> None:
    """Add the new language to LANGUAGE_DISPLAY_NAMES in config.py if not already present.

    Locates the dict by searching for its opening line, then finds the closing
    brace and inserts a new entry before it.
    """
    if not config_path.exists():
        logger.warning(f"{config_path} not found — skipping LANGUAGE_DISPLAY_NAMES update")
        return

    lang_attr = re.sub(r"[^A-Za-z0-9]+", "_", language).upper().strip("_")
    new_key = f"Language.{lang_attr}"

    lines = config_path.read_text(encoding="utf-8").splitlines()

    # Idempotency check
    if any(new_key in line for line in lines):
        logger.info(f"{new_key} already present in LANGUAGE_DISPLAY_NAMES")
        return

    # Find the opening line of LANGUAGE_DISPLAY_NAMES
    dict_start: int | None = None
    for i, line in enumerate(lines):
        if re.search(r"^LANGUAGE_DISPLAY_NAMES\s*:", line):
            dict_start = i
            break

    if dict_start is None:
        logger.warning("Could not find LANGUAGE_DISPLAY_NAMES in config.py — skipping")
        return

    # Find the closing brace of the dict (first line that is just '}' after dict_start)
    close_idx: int | None = None
    for i in range(dict_start + 1, len(lines)):
        if lines[i].strip() == "}":
            close_idx = i
            break

    if close_idx is None:
        logger.warning("Could not find closing '}' of LANGUAGE_DISPLAY_NAMES — skipping")
        return

    indent = "    "
    lines.insert(close_idx, f'{indent}{new_key}: "{language_name}",')

    if dry_run:
        logger.info(f"[dry-run] would add {new_key}: {language_name!r} to LANGUAGE_DISPLAY_NAMES")
        return

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Added {new_key}: {language_name!r} to LANGUAGE_DISPLAY_NAMES in {config_path}")


def update_env_example(language: str, language_name: str, env_example_path: Path, dry_run: bool) -> None:
    """Patch .env.example to include the new language in EVA_LANGUAGE options.

    Idempotent: no-op if the language is already present.
    Uses text search rather than line numbers so it is robust to file growth.
    """
    if not env_example_path.exists():
        logger.warning(f"{env_example_path} not found — skipping .env.example update")
        return

    lines = env_example_path.read_text(encoding="utf-8").splitlines()
    prefix = _lang_to_env_prefix(language)

    # ── 1. Update the #e line for EVA_LANGUAGE ──────────────────────────────
    # Annotation lines appear BEFORE the variable definition, so we find the
    # '#v EVA_LANGUAGE=' line and then scan backwards for the '#e ' line.
    language_var_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip().startswith("#v EVA_LANGUAGE="):
            language_var_idx = i
            break

    enum_line_idx: int | None = None
    if language_var_idx is not None:
        for i in range(language_var_idx - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith("#e "):
                enum_line_idx = i
                break
            # Stop scanning back once we hit a blank line or an unrelated variable
            if not stripped or (stripped.startswith("#v ") or (not stripped.startswith("#") and "=" in stripped)):
                break

    if enum_line_idx is None:
        logger.warning("Could not find '#e' options line for EVA_LANGUAGE in .env.example — skipping enum update")
    else:
        existing_opts = [o.strip() for o in lines[enum_line_idx].split(" ", 1)[1].split(",") if o.strip()]
        # Use the base language code (e.g. 'es' from 'es-MX') for the enum option
        # because the #e list holds the values the selectbox will show.
        lang_code = language.lower()
        if lang_code not in existing_opts:
            existing_opts.append(lang_code)
            lines[enum_line_idx] = "#e " + ",".join(existing_opts)
            logger.info(f"Added '{lang_code}' to EVA_LANGUAGE options in .env.example")
        else:
            logger.info(f"'{lang_code}' already present in EVA_LANGUAGE options")

    # ── 2. Insert agent ID pair before "Default user simulator agents" ───────
    var_f = f"EVA_{prefix}_USER_F"
    var_m = f"EVA_{prefix}_USER_M"

    # Check idempotency
    existing_text = "\n".join(lines)
    if f"#v {var_f}=" in existing_text or f"{var_f}=" in existing_text:
        logger.info(f"{var_f} already present in .env.example — skipping agent ID insertion")
    else:
        # Find the anchor: first line that starts the "Language agent IDs" comment,
        # then advance past it to insert at the end of that subsection.
        anchor_idx: int | None = None
        for i, line in enumerate(lines):
            if re.search(r"#\s*-+\s*Language agent IDs", line):
                # Advance past the header line to insert after it
                anchor_idx = i + 1
                break

        if anchor_idx is None:
            logger.warning(
                "Could not find '# --- Language agent IDs ---' in .env.example — appending agent ID pair at end of file"
            )
            anchor_idx = len(lines)

        new_block = [
            f"#i ElevenLabs agent ID — {language_name}, female voice.",
            "#d string",
            "#x perturbation_mode=Language",
            f"#x EVA_LANGUAGE={language.lower()}",
            f"#v {var_f}=",
            "",
            f"#i ElevenLabs agent ID — {language_name}, male voice.",
            "#d string",
            "#x perturbation_mode=Language",
            f"#x EVA_LANGUAGE={language.lower()}",
            f"#v {var_m}=",
            "",
        ]
        lines[anchor_idx:anchor_idx] = new_block
        logger.info(f"Inserted {var_f} / {var_m} blocks into .env.example")

    if dry_run:
        logger.info("[dry-run] .env.example changes not written")
        return

    env_example_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Updated {env_example_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, help="BCP 47 language tag, e.g. 'fr', 'es-MX'")
    ap.add_argument("--language-name", required=True, help="Human-readable English name, e.g. 'French'")
    ap.add_argument(
        "--native-name",
        help="Optional native-script name shown in parentheses in the agent addendum, e.g. 'français'",
    )
    ap.add_argument("--domain", dest="domains", action="append", help="Domain (repeatable). Default: all.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--names-file", help="JSON file with male_first/female_first/last arrays")
    src.add_argument("--auto-generate-names", action="store_true")
    ap.add_argument("--dump-names", help="When auto-generating, also save the arrays here")
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-spelling-variation",
        action="store_true",
        help=(
            "Also generate a {lang}_spelling.json mapping for regional spelling variants "
            "(e.g. colour→color). Only needed for languages with significant orthographic "
            "divergence between dialects. English already ships one; most others don't need it."
        ),
    )
    ap.add_argument(
        "--record-id",
        help="Only mutate the matching record id (across all selected domains). Useful for inspecting a single-row diff.",
    )
    args = ap.parse_args()

    args.language = _normalize_lang(args.language)

    if args.language == "en":
        print("Refusing to overwrite 'en' — that is owned by migrate_to_culture_schema.py", file=sys.stderr)
        return 2

    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
