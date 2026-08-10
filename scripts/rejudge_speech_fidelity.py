"""Re-run the tts_fidelity judge against a corrupted speech-fidelity dataset.

Takes a dataset produced by corrupt_speech_fidelity_turns.py (real audio + a mix of
edited/clean intended_assistant_turns) and runs the actual tts_fidelity judge
(Gemini, per AudioJudgeMetric.default_model) against each record's real audio file
and its (possibly-edited) intended text, filling in a judge_1 field.

Corrupted turns should score 0 (the judge sees the edited text doesn't match the
real audio); untouched turns should stay 1. Records left entirely clean should
come back all-1s.

Resumable: a record with an existing non-errored judge_1 is skipped unless --force
or listed in --only-ids, so re-running after an interruption (or after editing a
few records) only re-judges what's missing/requested.

Usage:
    python scripts/rejudge_speech_fidelity.py \
        --dataset agent_speech_fidelity_test_set_fr.jsonl \
        --audio-dir agent_speech_fidelity_audios
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eva.metrics.base import MetricContext
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType
from eva.utils import router
from eva.utils.logging import get_logger, setup_logging

load_dotenv()
setup_logging(level="INFO")
logger = get_logger(__name__)

METRIC_NAME = "tts_fidelity"


def build_context(record: dict[str, Any], audio_dir: Path) -> tuple[MetricContext, bool]:
    """Build a minimal MetricContext — tts_fidelity only reads intended_assistant_turns,
    audio_assistant_path, and language/language_display_name (via context.language).
    Everything else is a harmless placeholder since this judge never touches it.

    Prefers the pre-generated "{audio_base}_trimmed.wav" (scripts/generate_trimmed_audio.py)
    over the full audio when it exists, so the judge sees byte-identical audio to what a
    human labeler hears in the app. Returns (context, used_pretrimmed) — callers must set
    metric.trim_silence = False when used_pretrimmed is True, otherwise the judge would
    re-trim an already-trimmed file.
    """
    audio_base = record["audio_id"].removesuffix(".wav")
    trimmed_path = audio_dir / f"{audio_base}_trimmed.wav"
    used_pretrimmed = trimmed_path.exists()
    audio_path = trimmed_path if used_pretrimmed else audio_dir / record["audio_id"]

    intended_turns = {int(tid): text for tid, text in record["intended_assistant_turns"].items()}
    context = MetricContext(
        record_id=str(record["original_id"]),
        user_goal="",
        user_persona="",
        expected_scenario_db={},
        initial_scenario_db={},
        final_scenario_db={},
        initial_scenario_db_hash="",
        final_scenario_db_hash="",
        agent_role="",
        agent_instructions="",
        agent_tools=[],
        agent_id="",
        current_date_time="",
        intended_assistant_turns=intended_turns,
        num_assistant_turns=len(intended_turns),
        audio_assistant_path=str(audio_path),
        language=record.get("language", "fr"),
        pipeline_type=PipelineType.CASCADE,
    )
    return context, used_pretrimmed


def is_judged(record: dict[str, Any]) -> bool:
    j = record.get("judge_1")
    if not j or j.get("error") is not None:
        return False
    ratings = (j.get("details") or {}).get("per_turn_ratings") or {}
    return any(v is not None for v in ratings.values())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the tts_fidelity judge against a speech-fidelity dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the speech-fidelity .jsonl dataset")
    parser.add_argument("--audio-dir", default="agent_speech_fidelity_audios")
    parser.add_argument(
        "--only-ids",
        nargs="*",
        default=None,
        help="original_ids to (re-)judge. If omitted, judges every record missing a valid judge_1.",
    )
    parser.add_argument("--force", action="store_true", help="Re-judge even records that already have a valid judge_1.")
    args = parser.parse_args()

    model_list_json = os.getenv("EVA_MODEL_LIST")
    if not model_list_json:
        sys.exit("Error: EVA_MODEL_LIST env var is required.")
    router.init(json.loads(model_list_json))

    registry = get_global_registry()
    if registry.get(METRIC_NAME) is None:
        # NOTE: list_metrics() deliberately filters out exclude_from_default_metrics=True
        # metrics (tts_fidelity is one, being diagnostic-only) — use get()/create() instead.
        sys.exit(f"Error: metric '{METRIC_NAME}' not found in registry.")

    dataset_path = Path(args.dataset)
    records = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    audio_dir = Path(args.audio_dir)
    only_ids = set(args.only_ids) if args.only_ids is not None else None

    n_judged = 0
    for record in records:
        oid = record["original_id"]
        if only_ids is not None and oid not in only_ids:
            continue
        if only_ids is None and not args.force and is_judged(record):
            logger.info(f"  skip (already judged): {oid}")
            continue

        context, used_pretrimmed = build_context(record, audio_dir)
        metric = registry.create(METRIC_NAME)
        if used_pretrimmed:
            metric.trim_silence = False  # already trimmed — don't re-trim an already-trimmed file
        score = await metric.compute(context)
        record["judge_1"] = score.model_dump(mode="json")
        n_judged += 1

        ratings = (score.details or {}).get("per_turn_ratings") or {}
        expected = record["expected_rating"]
        corrupted_turn = next(iter(record.get("corruption_log", {})), None)
        logger.info(
            f"{oid} (expected_rating={expected}, corrupted_turn={corrupted_turn}): "
            f"per_turn_ratings={ratings}  error={score.error}"
        )

    dataset_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    logger.info(f"Judged {n_judged} record(s). Wrote {len(records)} total records to {dataset_path}")

    # Quick sanity summary: did corrupted turns actually get flagged, did clean records stay clean?
    hits, misses = 0, 0
    for record in records:
        j1 = record.get("judge_1")
        if not j1:
            continue
        ratings = (j1.get("details") or {}).get("per_turn_ratings") or {}
        corrupted_turns = set(record.get("corruption_log", {}).keys())
        for tid, rating in ratings.items():
            expected_turn_rating = 0 if str(tid) in corrupted_turns else 1
            if rating == expected_turn_rating:
                hits += 1
            else:
                misses += 1
    if hits + misses:
        logger.info(f"Per-turn judge accuracy vs. planted corruption: {hits}/{hits + misses} turns matched expectation")


if __name__ == "__main__":
    asyncio.run(main())
