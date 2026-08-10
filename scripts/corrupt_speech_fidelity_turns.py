"""Retroactively corrupt intended-assistant-turn text for speech-fidelity calibration data.

Takes real audio-pipeline records pulled from an existing toolkit run (real audio,
real "intended" ground-truth text that originally matched that audio) and, for a
chosen subset of records, edits one or more turns so the (edited) "intended" text
no longer matches what the (unedited, real) audio actually says. Re-judging against
the edited text should then produce a genuine rating-0 for those turns, while
records left untouched should stay at rating-1 (audio still matches text).

Records can be corrupted three ways:

  1. Synthetic multi-turn corruption (--corrupt-ids): edit N turns per record via
     LLM, cycling entity/truncation/insertion. N defaults to a distribution sampled
     from Tara's original English dataset's own bad-turn-count histogram
     ({1:1, 2:6, 3:13, 4:9, 5:1} — mode 3, not always 1), so our "bad" records match
     her intensity rather than a single subtle edit.
  2. Natural-defect passthrough (--natural-defect-ids): some real recordings already
     have a genuine fidelity issue in the ORIGINAL (pre-edit) toolkit judge score,
     unrelated to anything we did. These go straight into the corrupted set with NO
     editing — expected_rating=0, intended_assistant_turns == original text, and a
     natural_defect_turns field recording which turns and why (from the original
     agent_speech_fidelity score already present in the toolkit run's metrics.json).
     This is a more authentic "bad" signal than synthetic editing, and free.
  3. Left alone entirely -> stays in the clean pool (expected_rating=1).

Three synthetic corruption types, chosen to align with the tts_fidelity judge's own
failure-mode taxonomy (configs/prompts/judge.yaml: entity_error, truncation,
garbled_hallucination, insertion_hallucination, wrong_language). Only three are
achievable by editing text alone, since the audio itself is real and fixed —
garbled_hallucination and wrong_language require actually distorting/re-recording
audio, not just editing text:

  - entity_error   — swap one entity value (code, date, amount, name).
  - truncation     — append one fabricated extra sentence the audio never said.
  - insertion      — remove one real sentence that the audio DOES say.

This does NOT touch anything under the pulled toolkit records directory — it only
reads from there. Nothing is overwritten: every output record carries BOTH the
edited intended_assistant_turns (what the app/judge will see) AND the original,
unedited text, plus a corruption_log entry per edited turn recording exactly what
was changed and which failure mode we expect the judge to tag.

Does not re-run the speech-fidelity judge — that's a separate step (rejudge_speech_fidelity.py).

Usage:
    python scripts/corrupt_speech_fidelity_turns.py \
        --records-root debug_output/toolkit_itsm_fr/records \
        --domain itsm \
        --out agent_speech_fidelity_test_set_fr.jsonl \
        --audio-dir agent_speech_fidelity_audios \
        --corrupt-ids 35 71 41 \
        --natural-defect-ids 29 23 46 \
        --only-ids 35 71 41 29 23 46   # resume: recompute only these, reuse the rest
"""

import argparse
import asyncio
import itertools
import json
import os
import random
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
EXPECTED_FAILURE_MODE = {
    "entity": "entity_error",
    "truncation": "truncation",
    "insertion": "insertion_hallucination",
}

# Sampled from Tara's original English agent_speech_fidelity_test_set.jsonl:
# distribution of #bad-turns across her 30 expected_rating=0 records was
# {1: 1, 2: 6, 3: 13, 4: 9, 5: 1} — mode 3, average ~3. We match that intensity
# rather than always corrupting exactly one turn.
BAD_TURN_COUNT_CHOICES = [1, 2, 3, 4, 5]
BAD_TURN_COUNT_WEIGHTS = [1, 6, 13, 9, 1]

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


def rank_candidate_turns(turns: dict[str, str], corruption_type: str) -> list[str]:
    """Rank turns best-first for a given corruption type, skipping turn 0 (usually a greeting).

    entity: turns with more digits ranked first (proxy for codes/amounts/dates) —
    still just a ranking, not a filter, since some entities are spelled out as words.
    truncation/insertion: turns without internal duplication (see
    has_internal_duplication) ranked first by length, then duplicated turns as a
    last resort.
    """
    candidates = {tid: text for tid, text in turns.items() if tid != "0"}
    if not candidates:
        return []
    if corruption_type == "entity":
        return sorted(candidates, key=lambda tid: len(re.findall(r"\d", candidates[tid])), reverse=True)

    clean_ids = [tid for tid in candidates if not has_internal_duplication(candidates[tid])]
    dup_ids = [tid for tid in candidates if tid not in clean_ids]
    clean_ids.sort(key=lambda tid: len(candidates[tid]), reverse=True)
    dup_ids.sort(key=lambda tid: len(candidates[tid]), reverse=True)
    return clean_ids + dup_ids


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


def load_saved_metrics(record_dir: Path) -> dict[str, Any]:
    metrics_path = record_dir / "metrics.json"
    return json.loads(metrics_path.read_text())


async def corrupt_multiple_turns(
    llm_client: LiteLLMClient,
    original_turns: dict[str, str],
    n_edits: int,
    original_id: str,
    type_start_index: int,
) -> dict[str, Any]:
    """Apply up to n_edits corruptions to distinct turns, cycling corruption types.

    Returns corruption_log (possibly with fewer than n_edits entries if candidates
    run out or the LLM declines every attempt for a slot — that's an accepted
    partial outcome, not an error, same philosophy as the single-turn version).
    """
    used_turns: set[str] = set()
    corruption_log: dict[str, Any] = {}
    type_cycle = itertools.islice(itertools.cycle(CORRUPTION_TYPES), type_start_index, None)

    for _ in range(n_edits):
        ctype = next(type_cycle)
        candidates = [tid for tid in rank_candidate_turns(original_turns, ctype) if tid not in used_turns]
        applied = False
        for target_tid in candidates[:4]:  # cap attempts per slot so one record can't run away
            result = await corrupt_turn(llm_client, original_turns[target_tid], ctype)
            if not (result and result.get("has_edit") and result.get("edited_text")):
                continue
            result = _reject_if_noop(result, original_turns[target_tid], ctype, original_id)
            if not (result and result.get("has_edit") and result.get("edited_text")):
                continue

            used_turns.add(target_tid)
            corruption_log[target_tid] = {
                "corruption_type": ctype,
                "expected_failure_mode": EXPECTED_FAILURE_MODE[ctype],
                "detail": {k: v for k, v in result.items() if k not in ("edited_text", "has_edit")},
                "original_text": original_turns[target_tid],
                "edited_text": result["edited_text"],
            }
            logger.info(
                f"{original_id} turn {target_tid}: [{ctype}] "
                f"{ {k: v for k, v in result.items() if k not in ('edited_text', 'has_edit')} }"
            )
            applied = True
            break
        if not applied:
            logger.warning(f"{original_id}: could not find a suitable turn for a {ctype} edit (slot skipped)")

    return corruption_log


def natural_defect_turns(saved_metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract the original (pre-edit) toolkit judge's bad turns, if any.

    Reads the toolkit run's own stored agent_speech_fidelity score (Tara's-era
    naming — confirmed elsewhere to be the same TTS-vs-text methodology as today's
    tts_fidelity) to find turns that were already imperfect in the real,
    unedited recording, with no editing on our part.
    """
    f = saved_metrics.get("metrics", {}).get("agent_speech_fidelity", {})
    if f.get("error"):
        return {}
    details = f.get("details") or {}
    ratings = details.get("per_turn_ratings") or {}
    explanations = details.get("per_turn_explanations") or {}
    return {
        tid: {"original_rating": rating, "original_explanation": explanations.get(tid, "")}
        for tid, rating in ratings.items()
        if rating != 1
    }


async def build_record(
    llm_client: LiteLLMClient,
    next_id: int,
    original_id: str,
    domain: str,
    record_dir: Path,
    audio_dir: Path,
    n_edits: int | None,
    type_start_index: int,
    is_natural_defect: bool,
) -> dict[str, Any] | None:
    """Build one dataset record (synthetically corrupted, natural-defect, or clean)
    and copy its audio into audio_dir.
    """
    saved = load_saved_metrics(record_dir)
    ctx = saved["context"]
    original_turns: dict[str, str] = ctx["intended_assistant_turns"]

    corruption_log: dict[str, Any] = {}
    natural_defects: dict[str, Any] = {}
    edited_turns = dict(original_turns)  # copy — never mutate the loaded original
    expected_rating = 1

    if is_natural_defect:
        natural_defects = natural_defect_turns(saved)
        if natural_defects:
            expected_rating = 0
            logger.info(f"{original_id}: natural defect, turns {sorted(natural_defects)} — no editing applied")
        else:
            logger.warning(f"{original_id}: marked as natural-defect but original judge score is clean — leaving as-is")
    elif n_edits is not None and n_edits > 0:
        corruption_log = await corrupt_multiple_turns(llm_client, original_turns, n_edits, original_id, type_start_index)
        if corruption_log:
            expected_rating = 0
            for tid, entry in corruption_log.items():
                edited_turns[tid] = entry["edited_text"]

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
        "natural_defect_turns": natural_defects,
    }


def sample_n_edits(rng: random.Random) -> int:
    return rng.choices(BAD_TURN_COUNT_CHOICES, weights=BAD_TURN_COUNT_WEIGHTS, k=1)[0]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Retroactively corrupt intended text for speech-fidelity data.")
    parser.add_argument("--records-root", required=True, help="Dir containing <original_id>/trial_0/metrics.json")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", required=True, help="Output .jsonl path")
    parser.add_argument("--audio-dir", default="agent_speech_fidelity_audios")
    parser.add_argument(
        "--corrupt-ids",
        nargs="*",
        default=[],
        help="original_ids to synthetically corrupt, as ID or ID:N (N = number of turns to edit). "
        "Bare IDs sample N from Tara's bad-turn-count distribution (mode 3). Records not listed here "
        "or in --natural-defect-ids are left clean (expected_rating=1).",
    )
    parser.add_argument(
        "--natural-defect-ids",
        nargs="*",
        default=[],
        help="original_ids whose ORIGINAL (pre-edit) toolkit judge score already has a real fidelity "
        "defect — added to the corrupted set with NO editing (expected_rating=0, "
        "natural_defect_turns populated from the original score).",
    )
    parser.add_argument("--model", default=CORRUPTION_MODEL)
    parser.add_argument("--seed", type=int, default=None, help="Seed for sampling N edits per record (default: unseeded)")
    parser.add_argument(
        "--only-ids",
        nargs="*",
        default=None,
        help="original_ids to actually (re)compute this run. Any other original_id already present "
        "in --out is reused verbatim (no LLM call, no audio recopy). If omitted, every record in "
        "--corrupt-ids/--natural-defect-ids/records-root is recomputed (original all-at-once behavior).",
    )
    args = parser.parse_args()

    model_list_json = os.getenv("EVA_MODEL_LIST")
    if not model_list_json:
        sys.exit("Error: EVA_MODEL_LIST env var is required.")
    router.init(json.loads(model_list_json))
    llm_client = LiteLLMClient(model=args.model)
    rng = random.Random(args.seed)

    records_root = Path(args.records_root)
    original_ids = sorted(
        (p.name for p in records_root.iterdir() if p.is_dir() and (p / "trial_0" / "metrics.json").exists()),
        key=lambda s: int(s) if s.isdigit() else s,
    )
    if not original_ids:
        sys.exit(f"Error: no records with trial_0/metrics.json found under {records_root}")

    natural_defect_ids = set(args.natural_defect_ids)
    n_edits_map: dict[str, int] = {}
    for i, entry in enumerate(args.corrupt_ids):
        if ":" in entry:
            oid, n_str = entry.split(":", 1)
            n_edits_map[oid] = int(n_str)
        else:
            n_edits_map[entry] = sample_n_edits(rng)

    overlap = natural_defect_ids & set(n_edits_map)
    if overlap:
        sys.exit(f"Error: ids in both --corrupt-ids and --natural-defect-ids: {overlap}")

    logger.info(f"{len(original_ids)} record(s) found.")
    logger.info(f"Synthetic corruption plan: {n_edits_map}")
    logger.info(f"Natural-defect passthrough: {sorted(natural_defect_ids)}")

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
    type_start_index = 0  # rotates across records so entity/truncation/insertion don't always start in lockstep

    for next_id, original_id in enumerate(original_ids, start=1):
        if only_ids is not None and original_id not in only_ids and original_id in existing_by_original_id:
            record = {**existing_by_original_id[original_id], "id": next_id}
            record.setdefault("natural_defect_turns", {})
            # BUG GUARD: "id" is positional (1..len(original_ids)) and can shift across runs
            # as records are added/removed elsewhere in the sorted list, even for records
            # that are themselves unchanged. audio_id is DERIVED from id, so a reused record
            # must have its audio filename resynced to match its (possibly new) id — otherwise
            # a freshly-computed record landing on the OLD id can silently overwrite this
            # reused record's audio file (same filename, different content). Always re-copy
            # from the pristine source rather than trust whatever's currently in audio_dir.
            canonical_audio_id = f"{next_id}_expected_rating_{record['expected_rating']}.wav"
            if record.get("audio_id") != canonical_audio_id:
                src_audio = records_root / original_id / "trial_0" / "audio_assistant.wav"
                if src_audio.exists():
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_audio, audio_dir / canonical_audio_id)
                    logger.info(f"  resync audio_id for {original_id}: {record.get('audio_id')} -> {canonical_audio_id}")
                    record["audio_id"] = canonical_audio_id
                else:
                    logger.warning(f"  {original_id}: reused record's source audio missing, cannot resync audio_id")
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
                n_edits=n_edits_map.get(original_id),
                type_start_index=type_start_index,
                is_natural_defect=original_id in natural_defect_ids,
            )
            if original_id in n_edits_map:
                type_start_index = (type_start_index + 1) % len(CORRUPTION_TYPES)
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

    # Self-heal: remove any audio file left over from a previous run's id numbering
    # that no longer matches any current record's audio_id.
    if audio_dir.exists():
        current_audio_ids = {r["audio_id"] for r in dataset}
        orphans = [p for p in audio_dir.iterdir() if p.is_file() and p.name not in current_audio_ids]
        for p in orphans:
            p.unlink()
        if orphans:
            logger.info(f"Removed {len(orphans)} orphaned audio file(s) left over from prior id numbering.")

    n_corrupted = sum(1 for r in dataset if r["expected_rating"] == 0)
    n_natural = sum(1 for r in dataset if r.get("natural_defect_turns"))
    n_synthetic = sum(1 for r in dataset if r["corruption_log"])
    by_type: dict[str, int] = {}
    for log in all_corruption_log.values():
        for detail in log.values():
            by_type[detail["corruption_type"]] = by_type.get(detail["corruption_type"], 0) + 1
    logger.info(
        f"Summary: {len(dataset)} total, {n_corrupted} corrupted (expected_rating=0) "
        f"[{n_natural} natural-defect, {n_synthetic} synthetic, edit-type counts {by_type}], "
        f"{len(dataset) - n_corrupted} clean (expected_rating=1)"
    )


if __name__ == "__main__":
    asyncio.run(main())
