"""Re-run a text judge N times over already-generated conversation traces.

This is stage 2 of the judge-calibration dataset pipeline. Stage 1
(``scripts/run_text_only.py``) generates conversation traces and a single judge
score per record (used as the selection signal for balancing the target-rating
distribution). This script takes those saved traces and re-runs the judge N
times each, capturing the judge's self-consistency, then assembles an
app-compatible dataset (same structure as ``eva_faithfulness_test_set.jsonl``:
``id``, ``original_id``, ``domain``, ``target``, ``conversation_trace``,
``judge_1`` .. ``judge_N``).

Because text judges do not pin ``temperature=0`` (see
``TextJudgeMetric.default_params``), the N runs vary naturally — that variance is
exactly what the human ``target``/labels are compared against.

Usage:
    # One or more Stage-1 run dirs, each tagged DIR:DOMAIN:TARGET
    EVA_MODEL_LIST="$EVA_MODEL_LIST" python scripts/rerun_judge.py \
        --metric faithfulness --n 3 \
        --out eva_faithfulness_test_set_fr.jsonl \
        --run debug_output/20260802_..._opus:airline:3 \
        --run debug_output/20260802_..._gpt-5.4:airline:2 \
        --run debug_output/20260802_..._gpt-4.1-mini:airline:1

Each --run points at a Stage-1 output directory (the ``{timestamp}_{model}``
dir); every ``{record_id}/metrics.json`` under it becomes one dataset record,
with ``target`` set to the tier's intended rating.

Notes:
    - Do NOT set JUDGE_MODEL: the judge must stay at its class default so all
      traces are scored by the same judge.
    - EVA_MODEL_LIST must be set (the judge model is resolved via the router).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from eva.metrics.base import MetricContext
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType
from eva.utils import router
from eva.utils.logging import get_logger, setup_logging

load_dotenv()
setup_logging(level="INFO")
logger = get_logger(__name__)


def context_from_saved(ctx: dict) -> MetricContext:
    """Rebuild a MetricContext from a saved metrics.json ``context`` dict.

    Only the fields a text judge reads are populated meaningfully; the
    deterministic-metric fields (scenario DBs, hashes) get harmless defaults
    because conversation-level text judges never touch them.
    """
    return MetricContext(
        record_id=ctx.get("record_id", ""),
        user_goal=ctx.get("user_goal", ""),
        user_persona=ctx.get("user_persona", ""),
        expected_scenario_db=ctx.get("expected_scenario_db") or {},
        initial_scenario_db=ctx.get("initial_scenario_db") or {},
        final_scenario_db=ctx.get("final_scenario_db") or {},
        initial_scenario_db_hash=ctx.get("initial_scenario_db_hash", ""),
        final_scenario_db_hash=ctx.get("final_scenario_db_hash", ""),
        agent_role=ctx.get("agent_role", ""),
        agent_instructions=ctx.get("agent_instructions", ""),
        agent_tools=ctx.get("agent_tools") or [],
        agent_id=ctx.get("agent_id", ""),
        current_date_time=ctx.get("current_date_time", ""),
        conversation_trace=ctx.get("conversation_trace") or [],
        num_turns=ctx.get("num_turns", 0) or 0,
        language=ctx.get("language", "en") or "en",
        pipeline_type=PipelineType(ctx.get("pipeline_type", "cascade") or "cascade"),
    )


async def run_judge_n_times(metric_name: str, context: MetricContext, n: int) -> list[dict]:
    """Run the named judge ``n`` times on one context. Returns N serialized MetricScores.

    A fresh metric instance is created per run so no state carries over between
    calls. Each call is an independent judge sample of the same trace.
    """
    registry = get_global_registry()
    scores: list[dict] = []
    for i in range(1, n + 1):
        metric = registry.create(metric_name)
        if metric is None:
            sys.exit(f"Error: metric '{metric_name}' not found in registry.")
        score = await metric.compute(context)
        rating = score.details.get("rating") if score.details else None
        logger.info(f"  [{context.record_id}] judge_{i}: rating={rating} score={score.score}")
        scores.append(score.model_dump(mode="json"))
    return scores


def discover_records(run_dir: Path) -> list[tuple[str, Path]]:
    """Return (record_id, metrics.json path) for every record under a Stage-1 run dir.

    Skips archived retry dirs (``*_failed_attempt_*``) so only the kept attempt
    of each record is judged.
    """
    found: list[tuple[str, Path]] = []
    for metrics_path in sorted(run_dir.rglob("metrics.json")):
        rec_dir = metrics_path.parent
        if "_failed_attempt_" in rec_dir.name:
            continue
        # record_id is the dir name (or parent/name for trial_* layouts)
        record_id = rec_dir.name
        if record_id.startswith("trial_"):
            record_id = rec_dir.parent.name
        found.append((record_id, metrics_path))
    return found


def parse_run_arg(raw: str) -> tuple[Path, str, int]:
    """Parse a --run value of the form DIR:DOMAIN:TARGET."""
    parts = raw.rsplit(":", 2)
    if len(parts) != 3:
        sys.exit(f"Error: --run must be DIR:DOMAIN:TARGET, got '{raw}'")
    run_dir, domain, target = parts
    p = Path(run_dir)
    if not p.exists():
        sys.exit(f"Error: run dir does not exist: {p}")
    return p, domain, int(target)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run a text judge N times and assemble an app-compatible dataset.")
    parser.add_argument("--metric", default="faithfulness", help="Metric name in the registry (default: faithfulness)")
    parser.add_argument("--n", type=int, default=3, help="Judge runs per record (default: 3)")
    parser.add_argument("--out", required=True, help="Output .jsonl path (app-compatible dataset)")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR:DOMAIN:TARGET",
        help="A Stage-1 run dir tagged with its domain and intended target rating. Repeatable.",
    )
    parser.add_argument(
        "--make-labels",
        action="store_true",
        help="Also write an empty <metric>_labels_<lang>.json alongside --out for the labeling app.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge every trace even if --out already has a valid judge_1..judge_N entry for it "
        "(default: skip already-judged traces so re-runs only judge new/changed ones).",
    )
    args = parser.parse_args()

    model_list_json = os.getenv("EVA_MODEL_LIST")
    if not model_list_json:
        sys.exit("Error: EVA_MODEL_LIST env var is required (judge model is resolved via the router).")
    if os.getenv("JUDGE_MODEL"):
        logger.warning("JUDGE_MODEL is set — the judge will NOT use its class default. Unset it for calibration runs.")
    router.init(json.loads(model_list_json))

    runs = [parse_run_arg(r) for r in args.run]

    # Gather candidate traces, then dedup by (domain, target, original_id), keeping the
    # newest metrics.json. A resumed run that regenerated a record leaves multiple
    # timestamped dirs for it; without this we'd emit duplicate dataset records.
    best: dict[tuple, tuple] = {}
    for run_dir, domain, target in runs:
        for record_id, metrics_path in discover_records(run_dir):
            key = (domain, target, record_id)
            prev = best.get(key)
            if prev is None or metrics_path.stat().st_mtime > prev.stat().st_mtime:
                if prev is not None:
                    logger.info(f"  dedup {domain}/{record_id} t={target}: keeping newer trace")
                best[key] = metrics_path

    ordered = sorted(best.items(), key=lambda kv: kv[0])
    logger.info(f"{len(ordered)} unique trace(s) after dedup")

    # Load any existing dataset so already-judged traces are skipped by default.
    # Judges run at non-zero temperature, so blindly re-judging on every invocation
    # would silently overwrite (possibly already-labeled) judge scores.
    existing_by_key: dict[tuple, dict] = {}
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec.get("domain"), rec.get("target"), rec.get("original_id"))
                judge_keys = [f"judge_{i}" for i in range(1, args.n + 1)]
                complete = all(
                    rec.get(jk) and rec[jk].get("error") is None and (rec[jk].get("details") or {}).get("rating") is not None
                    for jk in judge_keys
                )
                if complete:
                    existing_by_key[key] = rec
        if existing_by_key:
            logger.info(f"Found {len(existing_by_key)} already-judged trace(s) in {out_path} — will skip these.")

    dataset: list[dict] = []
    for next_id, ((domain, target, record_id), metrics_path) in enumerate(ordered, start=1):
        key = (domain, target, record_id)
        reused = existing_by_key.get(key)
        if reused is not None:
            logger.info(f"  skip (already judged): {domain}/{record_id} t={target}")
            record = {**reused, "id": next_id}
            dataset.append(record)
            continue

        saved = json.loads(metrics_path.read_text())
        ctx_dict = saved.get("context") or {}
        context = context_from_saved(ctx_dict)
        judge_scores = await run_judge_n_times(args.metric, context, args.n)

        record = {
            "id": next_id,
            "original_id": record_id,
            "domain": domain,
            "target": target,
            "conversation_trace": ctx_dict.get("conversation_trace") or [],
        }
        for i, score in enumerate(judge_scores, start=1):
            record[f"judge_{i}"] = score
        dataset.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(dataset)} records to {out_path}")

    # Distribution summary: how many landed at each judge_1 rating vs their target.
    from collections import Counter

    by_target = Counter(r["target"] for r in dataset)
    logger.info(f"Target distribution: {dict(sorted(by_target.items()))}")
    matched = sum(1 for r in dataset if (r["judge_1"].get("details") or {}).get("rating") == r["target"])
    logger.info(f"judge_1 rating == target for {matched}/{len(dataset)} records")

    if args.make_labels:
        labels_path = out_path.with_name(out_path.stem.replace("_test_set", "") + "_labels.json")
        if not labels_path.exists():
            labels_path.write_text("{}\n", encoding="utf-8")
            logger.info(f"Created empty labels file: {labels_path}")
        else:
            logger.info(f"Labels file already exists, leaving untouched: {labels_path}")


if __name__ == "__main__":
    asyncio.run(main())
