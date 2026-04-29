"""Re-judge validation on existing trials and reclassify trial folders.

Re-runs a validation judge (default: user_behavioral_fidelity) on every trial
folder in a run — both clean (`trial_N`) and archived (`trial_N_failed_attempt_M`).
Pass/fail uses the same logic as `eva.orchestrator.validation_runner`. Trials
are then reclassified per (record, slot):

  * The chronologically-first passing folder becomes `trial_N`.
  * Other passing folders become `trial_N_extra_K`.
  * Still-failing folders become `trial_N_failed_attempt_M` (renumbered).

Conversation data is read from the embedded `context` dict in `metrics.json`,
so source files (audio/transcript/audit_log) are not required. The default
judge is text-only, so audio is not needed; `user_speech_fidelity` scores
are read from existing metrics.json without re-running.

Usage:
    python scripts/revalidate_and_promote.py --run-dir output/<run_id> [--dry-run]
"""

import argparse
import asyncio
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

import eva.metrics  # noqa: F401  -- triggers @register_metric for all metrics
from eva.metrics.base import MetricContext
from eva.metrics.registry import get_global_registry
from eva.models.config import PipelineType, RunConfig
from eva.models.results import MetricScore, RecordMetrics
from eva.orchestrator.validation_aggregates import compute_validation_aggregates
from eva.utils import router
from eva.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

GATE_METRIC = "conversation_valid_end"
SPEECH_METRIC = "user_speech_fidelity"
DEFAULT_JUDGE_METRIC = "user_behavioral_fidelity"
LLM_VALIDATION_METRICS = ("user_behavioral_fidelity", "user_speech_fidelity")

TRIAL_NAME_PATTERN = re.compile(r"^trial_(\d+)(?:_(failed_attempt|extra|unvalidated)_(\d+))?$")


@dataclass
class TrialFolder:
    record_id: str
    slot: int
    kind: str  # "clean" | "failed_attempt" | "unvalidated" | "extra"
    suffix_idx: int  # 0 for clean; M for _failed_attempt_M / _unvalidated_M; K for _extra_K
    path: Path
    chrono_key: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        # _unvalidated_M comes from prior runs that early-exited at this position;
        # treat it chronologically the same as _failed_attempt_M so order is stable
        # if it gets re-judged in a subsequent run.
        order = {"failed_attempt": 0, "unvalidated": 0, "clean": 1, "extra": 2}
        self.chrono_key = (order[self.kind], self.suffix_idx)


def _coerce_set(v) -> set:
    if isinstance(v, set):
        return v
    if isinstance(v, list):
        return set(v)
    return set()


def _coerce_int_keys(d):
    if not isinstance(d, dict):
        return d or {}
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def build_context_from_dict(ctx_dict: dict, record_id: str, agent_id: str = "") -> MetricContext:
    pipeline_raw = ctx_dict.get("pipeline_type", PipelineType.CASCADE)
    pipeline_type = PipelineType(pipeline_raw) if isinstance(pipeline_raw, str) else pipeline_raw

    return MetricContext(
        record_id=record_id,
        user_goal=ctx_dict.get("user_goal"),
        user_persona=ctx_dict.get("user_persona", ""),
        expected_scenario_db=ctx_dict.get("expected_scenario_db") or {},
        initial_scenario_db=ctx_dict.get("initial_scenario_db") or {},
        final_scenario_db=ctx_dict.get("final_scenario_db") or {},
        initial_scenario_db_hash=ctx_dict.get("initial_scenario_db_hash", ""),
        final_scenario_db_hash=ctx_dict.get("final_scenario_db_hash", ""),
        agent_role=ctx_dict.get("agent_role", ""),
        agent_instructions=ctx_dict.get("agent_instructions", ""),
        agent_tools=ctx_dict.get("agent_tools") or [],
        agent_id=ctx_dict.get("agent_id") or agent_id,
        current_date_time=ctx_dict.get("current_date_time", ""),
        num_turns=ctx_dict.get("num_turns", 0),
        num_tool_calls=ctx_dict.get("num_tool_calls", 0),
        tools_called=ctx_dict.get("tools_called") or [],
        conversation_ended_reason=ctx_dict.get("conversation_ended_reason"),
        duration_seconds=ctx_dict.get("duration_seconds", 0.0),
        output_dir=ctx_dict.get("output_dir", ""),
        audio_assistant_path=ctx_dict.get("audio_assistant_path"),
        audio_user_path=ctx_dict.get("audio_user_path"),
        audio_mixed_path=ctx_dict.get("audio_mixed_path"),
        transcribed_assistant_turns=_coerce_int_keys(ctx_dict.get("transcribed_assistant_turns")),
        transcribed_user_turns=_coerce_int_keys(ctx_dict.get("transcribed_user_turns")),
        intended_assistant_turns=_coerce_int_keys(ctx_dict.get("intended_assistant_turns")),
        intended_user_turns=_coerce_int_keys(ctx_dict.get("intended_user_turns")),
        audio_timestamps_assistant_turns=_coerce_int_keys(ctx_dict.get("audio_timestamps_assistant_turns")),
        audio_timestamps_user_turns=_coerce_int_keys(ctx_dict.get("audio_timestamps_user_turns")),
        num_assistant_turns=ctx_dict.get("num_assistant_turns") or 0,
        num_user_turns=ctx_dict.get("num_user_turns") or 0,
        tool_params=ctx_dict.get("tool_params") or [],
        tool_responses=ctx_dict.get("tool_responses") or [],
        conversation_trace=ctx_dict.get("conversation_trace") or [],
        latency_assistant_turns=_coerce_int_keys(ctx_dict.get("latency_assistant_turns")),
        assistant_interrupted_turns=_coerce_set(ctx_dict.get("assistant_interrupted_turns")),
        user_interrupted_turns=_coerce_set(ctx_dict.get("user_interrupted_turns")),
        pipeline_type=pipeline_type,
    )


def discover_trials(records_dir: Path) -> dict[tuple[str, int], list[TrialFolder]]:
    grouped: dict[tuple[str, int], list[TrialFolder]] = defaultdict(list)
    for record_dir in records_dir.iterdir():
        if not record_dir.is_dir():
            continue
        for trial_dir in record_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            m = TRIAL_NAME_PATTERN.match(trial_dir.name)
            if not m:
                continue
            slot = int(m.group(1))
            tag = m.group(2)
            if tag is None:
                kind, suffix_idx = "clean", 0
            else:
                kind, suffix_idx = tag, int(m.group(3))
            grouped[(record_dir.name, slot)].append(
                TrialFolder(record_id=record_dir.name, slot=slot, kind=kind, suffix_idx=suffix_idx, path=trial_dir)
            )
    for key in grouped:
        grouped[key].sort(key=lambda tf: tf.chrono_key)
    return grouped


def evaluate_pass(
    metrics: dict[str, MetricScore],
    thresholds: dict[str, float],
) -> tuple[bool, list[str]]:
    """Mirror eva.orchestrator.validation_runner.ValidationRunner._evaluate_record."""
    gate = metrics.get(GATE_METRIC)
    if gate is None or gate.error:
        return False, []
    gate_score = gate.normalized_score if gate.normalized_score is not None else gate.score
    if gate_score != 1.0:
        return False, []

    failed: list[str] = []
    for name in LLM_VALIDATION_METRICS:
        ms = metrics.get(name)
        if ms is None or ms.error:
            failed.append(name)
            continue
        if ms.skipped:
            continue
        score = ms.normalized_score if ms.normalized_score is not None else ms.score
        if name == SPEECH_METRIC and ms.details:
            per_turn = ms.details.get("per_turn_ratings", {})
            if any(r == 1 for r in per_turn.values() if r is not None):
                failed.append(name)
                continue
        threshold = thresholds.get(name, 1.0)
        if score is None or score < threshold:
            failed.append(name)
    return (not failed), failed


async def rejudge_one(
    folder: TrialFolder,
    metric_name: str,
    metric,
    thresholds: dict[str, float],
    semaphore: asyncio.Semaphore,
    agent_id: str = "",
    dry_run: bool = False,
) -> tuple[bool, bool, "MetricScore | None", "MetricScore | None"]:
    """Re-judge one folder. Returns (passed, llm_called, new_score, old_score).

    When dry_run=True, the judge still runs (so the preview reflects the new
    prompt) but metrics.json is not modified. `new_score` is the in-memory
    MetricScore from this run (or None if no judge call happened); `old_score`
    is the MetricScore that was on disk before this run (or None if absent).
    """
    metrics_path = folder.path / "metrics.json"
    if not metrics_path.exists():
        logger.warning(f"No metrics.json in {folder.path}; treating as fail")
        return False, False, None, None

    raw = json.loads(metrics_path.read_text())
    record_metrics = RecordMetrics(**raw)
    old_score = record_metrics.metrics.get(metric_name)

    # Skip judge call when the gate already failed — that folder can't pass.
    gate = record_metrics.metrics.get(GATE_METRIC)
    gate_score = None
    if gate is not None and not gate.error:
        gate_score = gate.normalized_score if gate.normalized_score is not None else gate.score
    if gate_score != 1.0:
        return False, False, None, old_score

    context_dict = record_metrics.context or {}
    if not context_dict:
        logger.warning(f"No context in {metrics_path}; cannot re-judge")
        passed, _ = evaluate_pass(record_metrics.metrics, thresholds)
        return passed, False, None, old_score

    context = build_context_from_dict(context_dict, folder.record_id, agent_id=agent_id)
    async with semaphore:
        try:
            new_score = await metric.compute(context)
        except Exception as e:
            logger.error(f"Judge failed for {folder.path}: {e}")
            return False, True, None, old_score

    record_metrics.metrics[metric_name] = new_score
    if not dry_run:
        metrics_path.write_text(record_metrics.model_dump_json(indent=2))

    passed, failed_metrics = evaluate_pass(record_metrics.metrics, thresholds)
    prefix = "[DRY] " if dry_run else ""
    logger.info(f"{prefix}{folder.record_id}/{folder.path.name}: passed={passed} failed={failed_metrics}")
    return passed, True, new_score, old_score


async def process_slot_rejudge(
    folders: list[TrialFolder],
    slot: int,
    metric_name: str,
    metric,
    thresholds: dict[str, float],
    semaphore: asyncio.Semaphore,
    agent_id: str = "",
    dry_run: bool = False,
) -> tuple[list[tuple[Path, Path]], dict, dict]:
    """Re-judge `_failed_attempt`s for one slot in chrono order, early-exit on first pass.

    Clean and existing extras are NOT re-judged — they already passed previously
    and the only expected change to the prompt is relaxation. If a failed_attempt
    now passes, it is promoted to `trial_N` and the original clean is demoted to
    `trial_N_extra_<next>`. Later failed_attempts are left untouched (unjudged).
    Returns (rename_plan, stats).
    """
    failed_attempts = sorted(
        [tf for tf in folders if tf.kind == "failed_attempt"],
        key=lambda tf: tf.suffix_idx,
    )
    clean = next((tf for tf in folders if tf.kind == "clean"), None)
    existing_extras = sorted(
        [tf for tf in folders if tf.kind == "extra"],
        key=lambda tf: tf.suffix_idx,
    )

    stats = {"llm_calls": 0, "gate_skipped": 0, "promoted": False, "rejudged": 0}

    chosen: TrialFolder | None = None
    chosen_new_score: MetricScore | None = None
    chosen_old_score: MetricScore | None = None
    rejudged_paths: list[str] = []
    chosen_index = -1
    for idx, tf in enumerate(failed_attempts):
        passed, called, new_score, old_score = await rejudge_one(
            tf, metric_name, metric, thresholds, semaphore, agent_id=agent_id, dry_run=dry_run
        )
        stats["rejudged"] += 1
        rejudged_paths.append(tf.path.name)
        if called:
            stats["llm_calls"] += 1
        else:
            stats["gate_skipped"] += 1
        if passed:
            chosen = tf
            chosen_new_score = new_score
            chosen_old_score = old_score
            chosen_index = idx
            break

    plan: list[tuple[Path, Path]] = []
    promotion: dict = {}
    if chosen is None:
        return plan, stats, promotion

    parent = chosen.path.parent
    promoted_str = f"{chosen.path.name}->trial_{slot}"
    plan.append((chosen.path, parent / f"trial_{slot}"))
    demoted_str = None
    if clean is not None:
        next_extra = max((tf.suffix_idx for tf in existing_extras), default=0) + 1
        demoted_name = f"trial_{slot}_extra_{next_extra}"
        demoted_str = f"{clean.path.name}->{demoted_name}"
        plan.append((clean.path, parent / demoted_name))

    # Rename the failed_attempts after `chosen` (early-exit skipped) to _unvalidated_M.
    unvalidated_renames: list[str] = []
    for j, tf in enumerate(failed_attempts[chosen_index + 1 :], start=1):
        new_name = f"trial_{slot}_unvalidated_{j}"
        plan.append((tf.path, parent / new_name))
        unvalidated_renames.append(f"{tf.path.name}->{new_name}")
    stats["promoted"] = True

    # Capture old + new judge details (in-memory; works in dry-run too).
    def _judge_summary(ms: MetricScore | None) -> dict:
        if ms is None:
            return {}
        return {
            "score": ms.score,
            "normalized_score": ms.normalized_score,
            "corruption_analysis": (ms.details or {}).get("corruption_analysis"),
        }

    old_details = _judge_summary(chosen_old_score)
    new_details = _judge_summary(chosen_new_score)

    promotion = {
        "record_id": chosen.record_id,
        "trial": slot,
        "promoted": promoted_str,
        "demoted": demoted_str,
        "marked_unvalidated": unvalidated_renames,
        "rejudged_in_order": rejudged_paths,
        "old_judge": old_details,
        "new_judge": new_details,
    }
    return plan, stats, promotion


def reclassify_slot_cached(
    folders: list[TrialFolder],
    pass_results: dict[Path, bool],
    slot: int,
) -> list[tuple[Path, Path]]:
    """Cached-mode reclassification (used with --no-rejudge).

    Picks chronologically-first passing folder as new clean, others passing as
    extras, others failing as renumbered failed_attempts.
    """
    parent = folders[0].path.parent
    passing = [tf for tf in folders if pass_results.get(tf.path, False)]
    failing = [tf for tf in folders if not pass_results.get(tf.path, False)]

    plan: list[tuple[Path, Path]] = []
    if not passing:
        for i, tf in enumerate(failing, start=1):
            plan.append((tf.path, parent / f"trial_{slot}_failed_attempt_{i}"))
        return plan

    chosen = passing[0]
    extras = passing[1:]
    plan.append((chosen.path, parent / f"trial_{slot}"))
    for i, tf in enumerate(extras, start=1):
        plan.append((tf.path, parent / f"trial_{slot}_extra_{i}"))
    for i, tf in enumerate(failing, start=1):
        plan.append((tf.path, parent / f"trial_{slot}_failed_attempt_{i}"))
    return plan


def apply_renames(plan: list[tuple[Path, Path]], dry_run: bool) -> int:
    actual = [(o, n) for o, n in plan if o != n]
    if dry_run:
        for old, new in actual:
            print(f"[DRY-RUN] {old.parent.name}/{old.name}  →  {new.name}")
        return len(actual)

    # Two-phase rename: temp names first to avoid collisions when names swap.
    temp_pairs: list[tuple[Path, Path]] = []
    for i, (old, new) in enumerate(actual):
        temp = old.parent / f".__revalidate_tmp_{i}__{old.name}"
        shutil.move(str(old), str(temp))
        temp_pairs.append((temp, new))
    for temp, new in temp_pairs:
        shutil.move(str(temp), str(new))
    return len(actual)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metric", default=DEFAULT_JUDGE_METRIC, help="Validation judge to re-run")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print rename plan only")
    parser.add_argument(
        "--no-rejudge",
        action="store_true",
        help="Skip judge re-run; reclassify using cached metrics.json scores",
    )
    parser.add_argument("--records", nargs="+", help="Restrict to these record IDs")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    config = json.loads((run_dir / "config.json").read_text())
    thresholds = config.get("validation_thresholds") or {}
    k = config.get("num_trials", 1)

    # Use the saved run's model_list (preserves the original deployment names like
    # `gpt-5.2-medium`). The api_keys were redacted to "***" on save; clear them so
    # LiteLLM falls back to the standard provider env vars (OPENAI_API_KEY, etc.).
    saved_model_list = config.get("model_list") or []
    for m in saved_model_list:
        params = m.get("litellm_params", {})
        if isinstance(params.get("api_key"), str) and set(params["api_key"]) <= {"*"}:
            params["api_key"] = None
    if not saved_model_list:
        saved_model_list = RunConfig().model_list  # fallback to env
    router.init(saved_model_list)

    # Load agent_id from the run's agent yaml (saved contexts predate this field)
    agent_config_path = Path(config.get("agent_config_path", ""))
    if not agent_config_path.is_absolute():
        agent_config_path = Path.cwd() / agent_config_path
    agent_id = ""
    if agent_config_path.exists():
        agent_id = (yaml.safe_load(agent_config_path.read_text()) or {}).get("id", "")
    logger.info(f"k={k}  thresholds={thresholds}  metric={args.metric}  agent_id={agent_id!r}")

    grouped = discover_trials(run_dir / "records")
    if args.records:
        wanted = set(args.records)
        grouped = {key: v for key, v in grouped.items() if key[0] in wanted}
    n_records = len({key[0] for key in grouped})
    n_folders = sum(len(v) for v in grouped.values())
    logger.info(f"Found {n_folders} trial folders across {n_records} records")
    # Snapshot OLD pass/fail per folder (cached scores), before any rejudge writes.
    old_pass: dict[str, bool] = {}  # key: f"{record}/{folder_name}"
    for (record_id, _slot), folders in grouped.items():
        for tf in folders:
            mp = tf.path / "metrics.json"
            ok = False
            if mp.exists():
                try:
                    rm = RecordMetrics(**json.loads(mp.read_text()))
                    ok, _ = evaluate_pass(rm.metrics, thresholds)
                except Exception:
                    pass
            old_pass[f"{record_id}/{tf.path.name}"] = ok

    full_plan: list[tuple[Path, Path]] = []
    summary = {
        "trials_total": len(grouped),
        "trials_promoted": 0,
        "trials_unchanged": 0,
        "llm_calls": 0,
        "gate_skipped": 0,
        "folders_rejudged": 0,
    }

    promotions: list[dict] = []
    if args.no_rejudge:
        pass_results: dict[Path, bool] = {}
        for folders in grouped.values():
            for tf in folders:
                metrics_path = tf.path / "metrics.json"
                if not metrics_path.exists():
                    pass_results[tf.path] = False
                    continue
                rm = RecordMetrics(**json.loads(metrics_path.read_text()))
                passed, _ = evaluate_pass(rm.metrics, thresholds)
                pass_results[tf.path] = passed
        for (record_id, slot), folders in sorted(grouped.items()):
            plan = reclassify_slot_cached(folders, pass_results, slot)
            full_plan.extend(plan)
            promoted_old = next((old for old, new in plan if new.name == f"trial_{slot}"), None)
            if promoted_old is not None:
                summary["trials_promoted"] += 1
                if promoted_old.name != f"trial_{slot}":
                    demoted_new = next((new.name for old, new in plan if old.name == f"trial_{slot}"), None)
                    promotions.append(
                        {
                            "record_id": record_id,
                            "trial": slot,
                            "promoted": f"{promoted_old.name}->trial_{slot}",
                            "demoted": f"trial_{slot}->{demoted_new}" if demoted_new else None,
                        }
                    )
            else:
                summary["trials_unchanged"] += 1
    else:
        registry = get_global_registry()
        metric = registry.create(args.metric, {})
        if metric is None:
            raise SystemExit(f"Metric '{args.metric}' not registered")
        sem = asyncio.Semaphore(args.max_concurrent)
        results = await asyncio.gather(
            *[
                process_slot_rejudge(
                    folders,
                    slot,
                    args.metric,
                    metric,
                    thresholds,
                    sem,
                    agent_id=agent_id,
                    dry_run=args.dry_run,
                )
                for (record_id, slot), folders in sorted(grouped.items())
            ]
        )
        for plan, stats, promotion in results:
            full_plan.extend(plan)
            summary["llm_calls"] += stats["llm_calls"]
            summary["gate_skipped"] += stats["gate_skipped"]
            summary["folders_rejudged"] += stats["rejudged"]
            if stats["promoted"]:
                summary["trials_promoted"] += 1
                if promotion:
                    promotions.append(promotion)
            else:
                summary["trials_unchanged"] += 1

    n_changed = apply_renames(full_plan, args.dry_run)
    logger.info(f"Renames {'planned' if args.dry_run else 'applied'}: {n_changed}")
    logger.info(f"Summary: {summary}")

    # Build new_pass map: for un-rejudged folders use cached old_pass; for promoted
    # folders use True; the demoted clean stays True (it was passing before and we
    # didn't re-judge it).
    new_pass = dict(old_pass)
    for promo in promotions:
        promoted_from = promo["promoted"].split("->", 1)[0]
        new_pass[f"{promo['record_id']}/{promoted_from}"] = True

    # Per-record before/after passing-slot counts.
    by_record_before: dict[str, int] = defaultdict(int)
    by_record_after: dict[str, int] = defaultdict(int)
    for (record_id, _), folders in grouped.items():
        any_old = any(old_pass.get(f"{record_id}/{tf.path.name}", False) for tf in folders)
        any_new = any(new_pass.get(f"{record_id}/{tf.path.name}", False) for tf in folders)
        if any_old:
            by_record_before[record_id] += 1
        if any_new:
            by_record_after[record_id] += 1
    record_changes = sorted(
        [
            {
                "record_id": rid,
                "passing_trials_before": by_record_before.get(rid, 0),
                "passing_trials_after": by_record_after.get(rid, 0),
            }
            for rid in {key[0] for key in grouped}
            if by_record_before.get(rid, 0) != by_record_after.get(rid, 0)
        ],
        key=lambda r: r["record_id"],
    )

    report = {
        "run_dir": str(run_dir),
        "metric": args.metric,
        "agent_id": agent_id,
        "k": k,
        "thresholds": thresholds,
        "dry_run": args.dry_run,
        "no_rejudge": args.no_rejudge,
        "summary": summary,
        "totals": {
            "passing_trials_before": sum(by_record_before.values()),
            "passing_trials_after": sum(by_record_after.values()),
            "records_with_full_k_before": sum(1 for v in by_record_before.values() if v >= k),
            "records_with_full_k_after": sum(1 for v in by_record_after.values() if v >= k),
        },
        "promotions": promotions,
        "record_changes": record_changes,
    }
    report_name = "revalidation_report.json" if not args.dry_run else "revalidation_report_dryrun.json"
    report_path = run_dir / report_name
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report: {report_path}")
    logger.info(
        f"Passing trials: before={report['totals']['passing_trials_before']} "
        f"after={report['totals']['passing_trials_after']}; "
        f"records with all {k} trials passing: "
        f"before={report['totals']['records_with_full_k_before']} "
        f"after={report['totals']['records_with_full_k_after']}"
    )

    if not args.dry_run:
        eval_summary_path = run_dir / "evaluation_summary.json"
        if eval_summary_path.exists():
            eval_summary = json.loads(eval_summary_path.read_text())
            eval_summary["validation_aggregates"] = compute_validation_aggregates(run_dir, thresholds)
            eval_summary_path.write_text(json.dumps(eval_summary, indent=2))
            logger.info(f"Updated validation_aggregates in {eval_summary_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
