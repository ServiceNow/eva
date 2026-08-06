"""Retroactively corrupt intended-assistant-turn text for speech-fidelity calibration data.

Takes real audio-pipeline records pulled from an existing toolkit run (real audio,
real "intended" ground-truth text that originally matched that audio) and, for a
chosen subset of records, makes one small text edit to a turn so the (edited)
"intended" text no longer matches what the (unedited, real) audio actually says.
Re-judging against the edited text should then produce a genuine rating-0 for that
turn, while records left untouched should stay at rating-1 (audio still matches text).

Three corruption types, chosen to align with the tts_fidelity judge's own failure-mode
taxonomy (configs/prompts/judge.yaml: entity_error, truncation, garbled_hallucination,
insertion_hallucination, wrong_language). Only three are achievable by editing text
alone, since the audio itself is real and fixed — garbled_hallucination and
wrong_language require actually distorting/re-recording audio, not just editing text:

  - entity_error   — swap one entity value (code, date, amount, name). The audio still
                     says the original value, so it no longer matches the (wrong) text.
  - truncation     — append one fabricated extra sentence the audio never said. From the
                     judge's view, expected text content is "missing" from the audio.
  - insertion      — remove one real sentence that the audio DOES say. From the judge's
                     view, the audio now contains "unscripted" content not in the text.

This does NOT touch anything under the pulled toolkit records directory — it only
reads from there. Nothing is overwritten: every output record carries BOTH the
edited intended_assistant_turns (what the app/judge will see) AND the original,
unedited text for that record, plus a corruption_log entry recording exactly what
was changed, which corruption type was used, and the failure mode we expect the
judge to tag if it re-scores correctly.

Does not re-run the speech-fidelity judge — that's a separate step, since it needs
the actual audio bytes and the AudioJudgeMetric machinery. This script only
produces the edited dataset + a change log; judge_1 is filled in afterward.

Usage:
    # default: cycle entity_error / truncation / insertion across --corrupt-ids
    python scripts/corrupt_speech_fidelity_turns.py \
        --records-root debug_output/toolkit_itsm_fr/records \
        --domain itsm \
        --out agent_speech_fidelity_test_set_fr.jsonl \
        --audio-dir agent_speech_fidelity_audios \
        --corrupt-ids 1 10 20 30 40 52 65 78 90 101

    # pin specific types per id: ID:TYPE (type in entity|truncation|insertion)
    python scripts/corrupt_speech_fidelity_turns.py ... \
        --corrupt-ids 1:entity 10:truncation 20:insertion
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eva.assistant.services.llm import LiteLLMClient
from eva.utils import router
from eva.utils.json_utils import extract_and_load_json_iter
from eva.utils.logging import get_logger, setup_logging

load_dotenv()
setup_logging(level="INFO")
logger = get_logger(__name__)

CORRUPTION_MODEL = "gpt-5.4"
CORRUPTION_TYPES = ["entity", "truncation", "insertion"]
# Maps our corruption type -> the tts_fidelity judge's own failure_mode label,
# so the log records what we'd expect a correct judge to tag.
EXPECTED_FAILURE_MODE = {
    "entity": "entity_error",
    "truncation": "truncation",
    "insertion": "insertion_hallucination",
}

_ENTITY_PROMPT = """You are helping build test data for a text-to-speech fidelity judge. You will \
be given one turn of a French voice-assistant conversation. Your job is to make exactly ONE small, \
surgical edit that changes a single TTS-critical entity to a different, still-plausible value — \
mimicking a realistic production bug where the text-to-speech script drifted from what was actually \
recorded (e.g. an off-by-one digit in a code, a swapped number, a changed date, a different name).

Entity categories to consider (pick exactly one occurrence to change):
- Alphanumeric codes / reference IDs (e.g. employee ID, ticket number)
- Phone number digits
- Dates or times
- Dollar amounts or other quantities
- Personal names

Rules:
- Change ONLY that one value. Do not rephrase, reword, or "improve" anything else in the turn.
- The new value must be the same type/format as the original (e.g. same number of digits for a code),
  but a genuinely different value — not a trivial reformatting of the same value.
- If the turn has no suitable entity to change, say so — do not force an edit onto unsuitable text.

## Turn text (French)
{turn_text}

## Response format
Respond with a single JSON object:
{{
  "has_edit": <bool: true if you made an edit, false if no suitable entity was found>,
  "original_value": "<string: the exact substring you changed, or empty string>",
  "new_value": "<string: the exact substring you replaced it with, or empty string>",
  "edited_text": "<string: the full turn text with only that one substring changed; \
identical to the input text if has_edit is false>"
}}
"""

_TRUNCATION_PROMPT = """You are helping build test data for a text-to-speech fidelity judge. You will \
be given one turn of a French voice-assistant conversation. Your job is to APPEND one short, plausible \
additional sentence to the end of this turn — something a script-generation bug might have inserted \
that was never actually recorded/spoken. This simulates a real production bug where the intended-text \
template includes a sentence that the TTS pass never rendered into audio.

Rules:
- The added sentence must fit naturally as a continuation of the turn's topic (same conversation,
  same policy/context) but must add new information not already present elsewhere in the turn
  (e.g. a new instruction, a new detail, a follow-up question) — not just a rephrasing of existing content.
- Do not change any existing text in the turn — only append.
- Keep the added sentence short (one sentence, not a paragraph).
- If the turn is a poor fit for this (e.g. too short, already ends with a hard cutoff/interruption tag
  like "[speaker likely cut itself off]"), say so — do not force an addition onto unsuitable text.

## Turn text (French)
{turn_text}

## Response format
Respond with a single JSON object:
{{
  "has_edit": <bool: true if you appended a sentence, false if unsuitable>,
  "added_sentence": "<string: exactly the sentence you appended, or empty string>",
  "edited_text": "<string: the full turn text with the sentence appended; \
identical to the input text if has_edit is false>"
}}
"""

_INSERTION_PROMPT = """You are helping build test data for a text-to-speech fidelity judge. You will \
be given one turn of a French voice-assistant conversation. Your job is to REMOVE exactly one \
self-contained sentence or clause from this turn — something the audio still says, but that will no \
longer appear in the "intended" text after your edit. This simulates a real production bug where the \
intended-text record was truncated/edited after the audio was already recorded, so the audio contains \
content that the text no longer accounts for.

Rules:
- Remove exactly ONE self-contained sentence or clause (not a single word, not the whole turn).
- The remaining text must still read as coherent, grammatical French after the removal.
- Do not change any other wording — only remove.
- Prefer removing a sentence that carries real content (an instruction, a fact, a question) — not just
  a filler phrase — so the removal is a meaningful, detectable gap.
- If the turn is too short to remove anything without destroying it (e.g. only one short sentence),
  say so — do not force a removal onto unsuitable text.

## Turn text (French)
{turn_text}

## Response format
Respond with a single JSON object:
{{
  "has_edit": <bool: true if you removed a sentence, false if unsuitable>,
  "removed_sentence": "<string: exactly the sentence/clause you removed, or empty string>",
  "edited_text": "<string: the full turn text with that sentence removed; \
identical to the input text if has_edit is false>"
}}
"""

_PROMPTS = {"entity": _ENTITY_PROMPT, "truncation": _TRUNCATION_PROMPT, "insertion": _INSERTION_PROMPT}


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation or newlines. Good enough for duplicate detection."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_sentence(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s).strip().lower()


def has_internal_duplication(text: str) -> bool:
    """True if any two sentences in this turn are near-duplicates of each other.

    These traces have a known pre-existing artifact where a trailing clause gets
    repeated verbatim within the same turn (a turn-grouping quirk upstream, not
    something we introduce). Removing or adding around a duplicated sentence is a
    no-op for the judge — the content is still present elsewhere in the turn — so
    such turns are unsuitable targets for truncation/insertion corruption.
    """
    sentences = [_normalize_sentence(s) for s in _split_sentences(text)]
    seen: set[str] = set()
    for s in sentences:
        if len(s) < 8:  # too short for duplication to be meaningful (e.g. "Bonjour")
            continue
        if s in seen:
            return True
        seen.add(s)
    return False


def pick_target_turn(turns: dict[str, str], corruption_type: str) -> str | None:
    """Pick a turn to corrupt, skipping turn 0 (usually just a greeting).

    entity: prefer the turn with the most digits (proxy for codes/amounts/dates).
    truncation/insertion: prefer the longest turn that does NOT have internally
    duplicated sentences (see has_internal_duplication) — more room to add/remove a
    clause without destroying coherence, and no risk of the edit being a no-op
    because the same content still lurks elsewhere in the turn. Falls back to the
    longest turn overall if every candidate has duplication.
    """
    candidates = {tid: text for tid, text in turns.items() if tid != "0"}
    if not candidates:
        return None
    if corruption_type == "entity":
        return max(candidates, key=lambda tid: len(re.findall(r"\d", candidates[tid])))

    clean_candidates = {tid: text for tid, text in candidates.items() if not has_internal_duplication(text)}
    pool = clean_candidates or candidates
    return max(pool, key=lambda tid: len(pool[tid]))


async def corrupt_turn(llm_client: LiteLLMClient, turn_text: str, corruption_type: str) -> dict[str, Any] | None:
    """Ask the LLM for one corruption edit of the given type. Returns the parsed response, or None on failure."""
    prompt = _PROMPTS[corruption_type].format(turn_text=turn_text)
    message, _ = await llm_client.complete([{"role": "user", "content": prompt}])
    response_text = getattr(message, "content", None) or (message if isinstance(message, str) else "")

    for obj, _ in extract_and_load_json_iter(response_text):
        if isinstance(obj, dict) and "edited_text" in obj:
            return obj
    logger.warning(f"Failed to parse {corruption_type} corruption response: {response_text[:300]}")
    return None


def _reject_if_noop(
    result: dict[str, Any], original_text: str, corruption_type: str, original_id: str
) -> dict[str, Any] | None:
    """Backstop against no-op edits: if the sentence the LLM added/removed is a
    near-duplicate of something still present elsewhere (edited text for insertion,
    original text for truncation), the edit doesn't actually create a text/audio
    mismatch — the content is still accounted for. Reject it (returns None) rather
    than silently accept a broken corruption.
    """
    key_field = "removed_sentence" if corruption_type == "insertion" else "added_sentence"
    changed_sentence = result.get(key_field, "")
    if not changed_sentence:
        return result

    norm_changed = _normalize_sentence(changed_sentence)
    if corruption_type == "insertion":
        edited_sentences = {_normalize_sentence(s) for s in _split_sentences(result["edited_text"])}
        if norm_changed in edited_sentences:
            logger.warning(
                f"{original_id}: rejected insertion edit — removed sentence still present "
                f"elsewhere in the turn (pre-existing duplication)"
            )
            return None
    elif corruption_type == "truncation":
        original_sentences = {_normalize_sentence(s) for s in _split_sentences(original_text)}
        if norm_changed in original_sentences:
            logger.warning(
                f"{original_id}: rejected truncation edit — added sentence already present in the original turn"
            )
            return None
    return result


def load_record_context(record_dir: Path) -> dict[str, Any]:
    metrics_path = record_dir / "metrics.json"
    saved = json.loads(metrics_path.read_text())
    return saved["context"]


async def build_record(
    llm_client: LiteLLMClient,
    next_id: int,
    original_id: str,
    domain: str,
    record_dir: Path,
    audio_dir: Path,
    corruption_type: str | None,
) -> dict[str, Any] | None:
    """Build one dataset record (corrupted or clean) and copy its audio into audio_dir."""
    ctx = load_record_context(record_dir)
    original_turns: dict[str, str] = ctx["intended_assistant_turns"]

    corruption_log: dict[str, Any] = {}
    edited_turns = dict(original_turns)  # copy — never mutate the loaded original
    expected_rating = 1

    if corruption_type is not None:
        target_tid = pick_target_turn(original_turns, corruption_type)
        if target_tid is None:
            logger.warning(f"{original_id}: no eligible turn found — leaving clean")
        else:
            result = await corrupt_turn(llm_client, original_turns[target_tid], corruption_type)
            if result and result.get("has_edit") and result.get("edited_text"):
                result = _reject_if_noop(result, original_turns[target_tid], corruption_type, original_id)

            if result and result.get("has_edit") and result.get("edited_text"):
                edited_turns[target_tid] = result["edited_text"]
                expected_rating = 0
                corruption_log[target_tid] = {
                    "corruption_type": corruption_type,
                    "expected_failure_mode": EXPECTED_FAILURE_MODE[corruption_type],
                    "detail": {k: v for k, v in result.items() if k not in ("edited_text", "has_edit")},
                    "original_text": original_turns[target_tid],
                    "edited_text": result["edited_text"],
                }
                logger.info(
                    f"{original_id} turn {target_tid}: [{corruption_type}] "
                    f"{ {k: v for k, v in result.items() if k not in ('edited_text', 'has_edit')} }"
                )
            else:
                logger.warning(f"{original_id}: turn {target_tid} unsuitable for {corruption_type} — leaving clean")

    audio_id = f"{next_id}_expected_rating_{expected_rating}.wav"
    src_audio = record_dir / "audio_assistant.wav"
    if not src_audio.exists():
        logger.error(f"{original_id}: missing audio_assistant.wav, skipping record")
        return None
    audio_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_audio, audio_dir / audio_id)

    return {
        "id": next_id,
        "original_id": original_id,
        "domain": domain,
        "intended_assistant_turns": edited_turns,
        "original_assistant_turns": original_turns,
        "expected_rating": expected_rating,
        "audio_id": audio_id,
        "corruption_log": corruption_log,
    }


def parse_corrupt_ids(raw: list[str]) -> dict[str, str]:
    """Parse --corrupt-ids entries of the form ID or ID:TYPE. Returns {original_id: corruption_type}."""
    result: dict[str, str] = {}
    for i, entry in enumerate(raw):
        if ":" in entry:
            oid, ctype = entry.split(":", 1)
            if ctype not in CORRUPTION_TYPES:
                sys.exit(f"Error: unknown corruption type '{ctype}' for id '{oid}' (use: {', '.join(CORRUPTION_TYPES)})")
        else:
            oid, ctype = entry, CORRUPTION_TYPES[i % len(CORRUPTION_TYPES)]
        result[oid] = ctype
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="Retroactively corrupt intended text for speech-fidelity data.")
    parser.add_argument("--records-root", required=True, help="Dir containing <original_id>/trial_0/metrics.json")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", required=True, help="Output .jsonl path")
    parser.add_argument("--audio-dir", default="agent_speech_fidelity_audios")
    parser.add_argument(
        "--corrupt-ids",
        nargs="*",
        default=None,
        help="original_ids to corrupt, as ID or ID:TYPE (type in entity|truncation|insertion). "
        "Bare IDs cycle through the three types in order. Records not listed are left clean "
        "(expected_rating=1). If omitted, the second half of the sorted record list is corrupted "
        "(cycling types).",
    )
    parser.add_argument("--model", default=CORRUPTION_MODEL)
    parser.add_argument(
        "--only-ids",
        nargs="*",
        default=None,
        help="original_ids to actually (re)compute this run. Any other original_id already present "
        "in --out is reused verbatim (no LLM call, no audio recopy). If omitted, every record in "
        "--corrupt-ids/records-root is recomputed (original all-at-once behavior).",
    )
    args = parser.parse_args()

    model_list_json = os.getenv("EVA_MODEL_LIST")
    if not model_list_json:
        sys.exit("Error: EVA_MODEL_LIST env var is required.")
    router.init(json.loads(model_list_json))
    llm_client = LiteLLMClient(model=args.model)

    records_root = Path(args.records_root)
    original_ids = sorted(
        (p.name for p in records_root.iterdir() if p.is_dir() and (p / "trial_0" / "metrics.json").exists()),
        key=lambda s: int(s) if s.isdigit() else s,
    )
    if not original_ids:
        sys.exit(f"Error: no records with trial_0/metrics.json found under {records_root}")

    if args.corrupt_ids is not None:
        corrupt_map = parse_corrupt_ids(args.corrupt_ids)
    else:
        half = len(original_ids) // 2
        corrupt_map = parse_corrupt_ids(original_ids[half:])

    logger.info(f"{len(original_ids)} record(s) found. Corrupting: {corrupt_map}")

    out_path = Path(args.out)
    existing_by_original_id: dict[str, dict[str, Any]] = {}
    if args.only_ids is not None and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_by_original_id[rec["original_id"]] = rec
        logger.info(f"Loaded {len(existing_by_original_id)} existing record(s) from {out_path} for reuse.")

    only_ids = set(args.only_ids) if args.only_ids is not None else None

    audio_dir = Path(args.audio_dir)
    dataset: list[dict[str, Any]] = []
    all_corruption_log: dict[str, Any] = {}

    for next_id, original_id in enumerate(original_ids, start=1):
        if only_ids is not None and original_id not in only_ids and original_id in existing_by_original_id:
            record = {**existing_by_original_id[original_id], "id": next_id}
            logger.info(f"  reuse (unchanged): {original_id}")
        else:
            record_dir = records_root / original_id / "trial_0"
            record = await build_record(
                llm_client=llm_client,
                next_id=next_id,
                original_id=original_id,
                domain=args.domain,
                record_dir=record_dir,
                audio_dir=audio_dir,
                corruption_type=corrupt_map.get(original_id),
            )
        if record is None:
            continue
        if record["corruption_log"]:
            all_corruption_log[original_id] = record["corruption_log"]
        dataset.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(dataset)} records to {out_path}")

    log_path = out_path.with_name(out_path.stem + "_corruption_log.json")
    log_path.write_text(json.dumps(all_corruption_log, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote corruption log to {log_path}")

    n_corrupted = sum(1 for r in dataset if r["expected_rating"] == 0)
    by_type = {}
    for log in all_corruption_log.values():
        for detail in log.values():
            by_type[detail["corruption_type"]] = by_type.get(detail["corruption_type"], 0) + 1
    logger.info(
        f"Summary: {len(dataset)} total, {n_corrupted} corrupted (expected_rating=0) [{by_type}], "
        f"{len(dataset) - n_corrupted} clean (expected_rating=1)"
    )


if __name__ == "__main__":
    asyncio.run(main())
