"""One-off repair: replace specific insertion-type corruptions with entity-type ones.

Context: the tts_fidelity judge's rating criteria only flags "added audio content"
as an error when it introduces a factually incorrect entity — plain unscripted
filler (a friendly sign-off, a confirmation) is never flagged. This made our
`insertion` corruption type almost undetectable by design (0/17 caught), unlike
`entity` (has a matching criterion) — confirmed by re-deriving Tara's own original
corruption mix from her judge explanations, where insertion-style edits are nearly
absent (1/76) for the same reason.

Decision: keep insertion in the 5 records where it's the ONLY edit (documented
evidence of the judge's blind spot), and replace it with an entity edit in the 9
records where it co-occurs with other edit types (12 insertion turns total).

For each targeted (record, turn):
    1. Revert that turn's text to the original (undoing the insertion edit).
    2. Remove that entry from corruption_log.
    3. Attempt an entity edit — first on the SAME turn, then on other unused turns
       in the record (excluding turns already claimed by other surviving edits in
       that record) — using the same corrupt_turn/_reject_if_noop logic as the main
       corruption script, for consistency.
    4. If no turn in the record accepts an entity edit, the turn is left reverted
       (record still has expected_rating=0 as long as another edit survives).

Does NOT touch audio — only intended_assistant_turns/corruption_log for the
targeted records. Prints which original_ids need re-judging afterward.

Usage:
    python scripts/replace_insertion_with_entity.py \
        --dataset agent_speech_fidelity_test_set_fr.jsonl \
        --records-root debug_output/toolkit_itsm_fr/records \
        --target-ids 25 35 38 41 42 75 77 88 91
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eva.assistant.services.llm import LiteLLMClient
from eva.utils import router
from eva.utils.logging import get_logger, setup_logging

load_dotenv()
setup_logging(level="INFO")
logger = get_logger(__name__)

# Reuse functions from the main corruption module without duplicating logic.
_spec = importlib.util.spec_from_file_location("corrupt_mod", Path(__file__).parent / "corrupt_speech_fidelity_turns.py")
corrupt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corrupt_mod)


async def replace_record(llm_client: LiteLLMClient, record: dict[str, Any], original_id: str) -> bool:
    """Revert this record's insertion turn(s) and try an entity edit instead. Returns True if changed."""
    log = record["corruption_log"]
    original_turns = record["original_assistant_turns"]
    insertion_turns = [tid for tid, e in log.items() if e["corruption_type"] == "insertion"]
    if not insertion_turns:
        return False

    other_used_turns = {tid for tid in log if tid not in insertion_turns}
    changed = False

    for tid in insertion_turns:
        # Step 1/2: revert
        record["intended_assistant_turns"][tid] = original_turns[tid]
        del log[tid]
        logger.info(f"{original_id} turn {tid}: reverted insertion edit")

        # Step 3: try entity edit, same turn first, then other unused candidates
        candidates = [tid] + [
            t for t in corrupt_mod.rank_candidate_turns(original_turns, "entity")
            if t != tid and t not in other_used_turns and t not in log
        ]
        applied = False
        for cand in candidates[:4]:
            result = await corrupt_mod.corrupt_turn(llm_client, original_turns[cand], "entity")
            if not (result and result.get("has_edit") and result.get("edited_text")):
                continue
            result = corrupt_mod._reject_if_noop(result, original_turns[cand], "entity", original_id)
            if not (result and result.get("has_edit") and result.get("edited_text")):
                continue
            record["intended_assistant_turns"][cand] = result["edited_text"]
            log[cand] = {
                "corruption_type": "entity",
                "expected_failure_mode": corrupt_mod.EXPECTED_FAILURE_MODE["entity"],
                "detail": {k: v for k, v in result.items() if k not in ("edited_text", "has_edit")},
                "original_text": original_turns[cand],
                "edited_text": result["edited_text"],
            }
            other_used_turns.add(cand)
            logger.info(
                f"{original_id} turn {cand}: [entity] "
                f"{ {k: v for k, v in result.items() if k not in ('edited_text', 'has_edit')} }"
            )
            applied = True
            changed = True
            break
        if not applied:
            logger.warning(f"{original_id}: could not find an entity substitute for reverted turn {tid}")

    return changed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Replace insertion-type corruptions with entity-type ones.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-ids", nargs="+", required=True, help="original_ids to fix")
    parser.add_argument("--model", default=corrupt_mod.CORRUPTION_MODEL)
    args = parser.parse_args()

    model_list_json = os.getenv("EVA_MODEL_LIST")
    if not model_list_json:
        sys.exit("Error: EVA_MODEL_LIST env var is required.")
    router.init(json.loads(model_list_json))
    llm_client = LiteLLMClient(model=args.model)

    dataset_path = Path(args.dataset)
    records = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    by_original_id = {r["original_id"]: r for r in records}

    target_ids = set(args.target_ids)
    missing = target_ids - set(by_original_id)
    if missing:
        sys.exit(f"Error: original_ids not found in dataset: {missing}")

    touched = []
    for oid in sorted(target_ids, key=lambda s: int(s) if s.isdigit() else s):
        record = by_original_id[oid]
        changed = await replace_record(llm_client, record, oid)
        if changed:
            touched.append(oid)
            # judge_1 is now stale for this record's edited turns — clear it so
            # rejudge_speech_fidelity.py's is_judged() treats it as needing rejudging.
            record["judge_1"] = None

    dataset_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    logger.info(f"Wrote {len(records)} records to {dataset_path}")
    logger.info(f"Records changed (need re-judging): {touched}")


if __name__ == "__main__":
    asyncio.run(main())
