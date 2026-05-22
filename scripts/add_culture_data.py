"""Add a language/culture to an already-migrated dataset.

For each record in ``data/<domain>_dataset.jsonl``:
  1. Pick a culturally appropriate (first, last) name pair matching the record's
     gender, deterministically seeded by record id + language.
  2. Set ``record.culture_overrides[<lang>] = {first_name, last_name}``.
  3. Translate ``user_goal.en_starting_utterance`` -> ``user_goal.<lang>_starting_utterance``
     via LLM. Placeholders ``<FIRST_NAME>`` / ``<LAST_NAME>`` in the source are
     preserved in the translation.

Also writes (or merges into) ``configs/agents/language_addenda.yaml`` with a short
"respond in <language>" instruction appended to the agent prompt at runtime.

Idempotent: a record is skipped if it already has ``culture_overrides[<lang>]``
and ``user_goal.<lang>_starting_utterance``.

Prerequisites:
  - Phase A migration has been run (records have ``culture_overrides.en`` and
    ``user_goal.en_starting_utterance``).
  - ``user_config.gender`` is present and one of ``man`` / ``woman`` for every record.

Name source (one of):
  --names-file path/to/names.json  containing
      {"male_first": [...], "female_first": [...], "last": [...]}
  --auto-generate-names   ask the LLM for 40+40+40 culturally authentic names
                          (mix of romanized and native-script).

Usage:
  python scripts/add_culture_data.py --domain airline --language fr \\
      --language-name French --auto-generate-names

  python scripts/add_culture_data.py --domain itsm --language es-MX \\
      --language-name "Mexican Spanish" --names-file es_mx_names.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from eva.utils.culture import FIRST_NAME_PLACEHOLDER, LAST_NAME_PLACEHOLDER
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
ADDENDA_PATH = REPO_ROOT / "configs" / "agents" / "language_addenda.yaml"
INITIAL_MESSAGES_PATH = REPO_ROOT / "configs" / "agents" / "initial_messages.yaml"

DEFAULT_MODEL = "gpt-5.2"
TRANSLATION_BATCH = 25

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


async def add_culture(
    domain: str,
    language: str,
    language_name: str,
    names: dict[str, list[str]],
    romanized_names: dict[str, list[str]],
    llm: LLMClient,
    dry_run: bool,
    addendum: str,
    record_id: str | None = None,
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


async def amain(args: argparse.Namespace) -> int:
    llm = LLMClient(model=args.llm_model, params={"temperature": 0.0})

    if args.auto_generate_names:
        logger.info(f"Generating {args.language_name} name arrays via LLM")
        names = await _generate_names(args.language_name, llm)
        if args.dump_names:
            Path(args.dump_names).write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Wrote generated names to {args.dump_names}")
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

    domains = args.domains or [p.stem.removesuffix("_dataset") for p in sorted(DATA_DIR.glob("*_dataset.jsonl"))]
    for domain in domains:
        logger.info(f"=== Domain: {domain} ===")
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
        )
    return 0


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
        "--record-id",
        help="Only mutate the matching record id (across all selected domains). Useful for inspecting a single-row diff.",
    )
    args = ap.parse_args()

    if args.language == "en":
        print("Refusing to overwrite 'en' — that is owned by migrate_to_culture_schema.py", file=sys.stderr)
        return 2

    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
