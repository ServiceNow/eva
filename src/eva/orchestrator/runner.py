"""Benchmark runner - main orchestrator for running voice agent benchmarks."""

import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from tqdm.asyncio import tqdm
from tqdm.std import Bar

from eva.metrics.legacy_aliases import rename_metric_keys, rename_metric_list
from eva.metrics.runner import MetricsRunner, MetricsRunResult
from eva.models.agents import AgentConfig
from eva.models.config import PipelineType, RunConfig
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, RunResult
from eva.orchestrator.port_pool import PortPool
from eva.orchestrator.validation_runner import ValidationResult, ValidationRunner
from eva.orchestrator.worker import ConversationWorker
from eva.utils.conversation_checks import check_conversation_finished, find_records_with_llm_generic_error
from eva.utils.culture import get_language_addendum
from eva.utils.logging import get_logger
from eva.utils.pass_at_k import ATTEMPT_SUFFIX_PATTERN, parse_trial_record_id
from eva.utils.provenance import capture_provenance, resolve_tool_module_file

logger = get_logger(__name__)

# Register a bright blue progress-bar colour.
Bar.COLOURS["BRIGHT_BLUE"] = "\x1b[94m"


class TqdmNewlineStream:
    """Stream proxy that writes newline-terminated lines rather than tqdm's in-place '\r' redraws."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, s: str) -> int:
        s = s.lstrip("\r")
        if s and not s.endswith("\n"):
            s += "\n"
        return self._stream.write(s)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class BenchmarkRunner:
    """Main orchestrator for running voice agent benchmarks.

    Manages:
    - Loading dataset and tool mocks
    - Port pool for parallel conversations
    - Rate limiting across providers
    - Spawning conversation workers
    - Collecting and aggregating results
    """

    def __init__(self, config: RunConfig):
        """Initialize the benchmark runner.

        Args:
            config: Run configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir) / config.run_id

        # Load agent config
        self.agent = self._load_agent_config()

        # Initialize port pool
        self.port_pool = PortPool(
            base_port=config.base_port,
            pool_size=config.port_pool_size,
        )

        # Per-metric configuration derived from run config (e.g. language for stt_wer).
        self._metric_configs: dict[str, dict] = {
            "stt_wer": {"language": config.language.value},
        }

        # Results tracking
        self._results: list[ConversationResult] = []
        self._failed_record_ids: list[str] = []

    def _load_agent_config(self) -> AgentConfig:
        """Load single agent configuration; append the language addendum once."""
        agent = AgentConfig.from_yaml(self.config.agent_config_path)
        addendum = get_language_addendum(self.config.language)
        if addendum:
            agent.instructions = f"{agent.instructions}\n\n{addendum}"
        return agent

    def _filter_records(self, records: list[EvaluationRecord]) -> list[EvaluationRecord]:
        """Filter records based on debug mode or record_ids.

        Args:
            records: All records from dataset

        Returns:
            Filtered list of records to run
        """
        # Debug mode takes precedence - run only 1 record
        if self.config.debug:
            logger.info("Debug mode enabled: running only 1 record")
            return records[:1]

        # Filter by specific record IDs if provided
        if self.config.record_ids:
            logger.info(f"Filtering to specific records: {self.config.record_ids}")
            filtered = [r for r in records if r.id in self.config.record_ids]

            # Warn if some IDs not found
            found_ids = {r.id for r in filtered}
            missing_ids = set(self.config.record_ids) - found_ids
            if missing_ids:
                logger.warning(f"Record IDs not found in dataset: {missing_ids}")

            return filtered

        # No filtering - return all records
        return records

    async def run(self, records: list[EvaluationRecord]) -> RunResult:
        """Run all records with validation and reruns.

        Single flat loop per output_id, up to max_rerun_attempts total:
        1. Run conversation
        2. Check conversation_finished — if not finished, retry
        3. Run validation metrics
        4. If validation passes → done; if not → retry
        5. After all attempts exhausted → record is failed
        6. Run full metrics on successful records only

        Args:
            records: List of evaluation records to run

        Returns:
            RunResult with final counts and duration
        """
        if not self.config.tool_module_path:
            self.config.tool_module_path = self.agent.tool_module_path
        try:
            tool_module_file = resolve_tool_module_file(self.config.tool_module_path)
            self.config.provenance = capture_provenance(self.config, tool_module_file=tool_module_file)
        except Exception as e:
            logger.warning(f"Failed to capture provenance: {e}")

        max_attempts = self.config.max_rerun_attempts
        logger.info(f"Starting benchmark with up to {max_attempts} attempts per record")

        # Apply record filtering (debug mode or specific record IDs)
        filtered_records = self._filter_records(records)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "records").mkdir(exist_ok=True)

        # Resolve exact models used (captures defaults from services.py + any alias labels)
        if self.config.model.pipeline_type == PipelineType.CASCADE:
            stt_params = self.config.model.stt_params
            tts_params = self.config.model.tts_params
            self.config.resolved_models = {
                "stt_provider": self.config.model.stt,
                "stt_model": stt_params["model"],
                "stt_alias": stt_params.get("alias"),
                "tts_provider": self.config.model.tts,
                "tts_model": tts_params["model"],
                "tts_alias": tts_params.get("alias"),
                "llm": self.config.model.llm,
            }

        config_path = self.output_dir / "config.json"
        config_data = self.config.model_dump(mode="json")
        pipeline_parts = self.config.model.pipeline_parts
        config_data["pipeline_parts"] = pipeline_parts
        config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False))

        # Build output_id list for tracking (supports pass@k)
        num_trials = self.config.num_trials
        output_id_to_record: dict[str, EvaluationRecord] = {}

        for record in filtered_records:
            if num_trials > 1:
                for trial_idx in range(num_trials):
                    oid = f"{record.id}/trial_{trial_idx}"
                    output_id_to_record[oid] = record
            else:
                output_id_to_record[record.id] = record

        all_output_ids = list(output_id_to_record.keys())
        pending_output_ids = list(all_output_ids)
        rerun_history: dict[str, list[dict]] = {}
        time_limit_attempt_counts: dict[str, int] = {}
        time_limit_validation_cache: dict[str, dict[int, ValidationResult]] = {}
        max_time_limit_attempts = int(self.config.validation_thresholds.get("max_time_limit_attempts", 1))
        time_limit_accepted_ids: set[str] = set()
        started_at = datetime.now()

        # Initialize port pool once before the attempt loop.
        await self.port_pool.initialize()

        # Semaphore limits concurrent conversations across all attempts.
        # All slots are always released after asyncio.gather, so reusing it is safe.
        semaphore = asyncio.Semaphore(self.config.max_concurrent_conversations)

        # Pre-create ValidationRunner for per-record pipelining. Its shared MetricsRunner
        # for validation metrics is lazily initialized on first use.
        _all_unique_records = list({id(r): r for r in output_id_to_record.values()}.values())
        pipeline_validation_runner = ValidationRunner(
            run_dir=self.output_dir,
            dataset=_all_unique_records,
            thresholds=self.config.validation_thresholds,
            metric_configs=self._metric_configs,
        )

        # Pre-create MetricsRunner so metrics can start as soon as records pass validation,
        # overlapping with reruns of failed records instead of waiting for ALL conversations.
        metrics_runner: MetricsRunner | None = None
        metrics_background_tasks: list[asyncio.Task] = []

        if self.config.metrics:
            all_unique_records = list({id(r): r for r in output_id_to_record.values()}.values())
            try:
                metrics_runner = MetricsRunner(
                    run_dir=self.output_dir,
                    dataset=all_unique_records,
                    metric_names=self.config.metrics,
                    metric_configs=self._metric_configs,
                    num_draws=self.config.num_trials,
                    force_rerun=self.config.force_rerun_metrics,
                )
            except Exception as e:
                logger.warning(f"Failed to pre-create MetricsRunner for overlapped execution: {e}")
                metrics_runner = None

        # LOOP: Run and validate, up to max_attempts total
        for attempt_number in range(1, max_attempts + 1):
            if not pending_output_ids:
                break

            # STEPS 1-3: Per-record pipeline — run, validate, and fire metrics as each
            # conversation completes rather than waiting for the full batch.
            # The semaphore slot is released before audio writes and before LLM validation
            # calls, so the next conversation can start earlier.

            async def _run_and_pipeline(
                record: EvaluationRecord, output_id: str
            ) -> tuple[str, ConversationResult | Exception, bool, ValidationResult | None]:
                try:
                    # Phase 1: Run conversation (bounded by semaphore)
                    async with semaphore:
                        result, audio_task = await self._run_conversation(record, output_id)

                    # result.json is now written by the worker itself (inside the
                    # semaphore) so it survives task cancellation and process signals.

                    # Phase 2: If the conversation didn't complete, skip validation.
                    # validate_one() handles the gate (conversation_valid_end); returning
                    # vr=None signals "not_finished" to the classification loop below.
                    if not (isinstance(result, ConversationResult) and result.completed):
                        return output_id, result, False, None

                    # Phase 4: Await audio task (needed for audio-based validation metrics)
                    if audio_task is not None:
                        try:
                            await audio_task
                        except Exception as e:
                            logger.warning(f"Audio save failed for {output_id}: {e}")

                    # Phase 5: Validate this record (gate + LLM metrics)
                    vr = await pipeline_validation_runner.validate_one(output_id)

                    # Phase 6: Fire metrics immediately if passed
                    if vr.passed and metrics_runner is not None:
                        rdir = self.output_dir / "records" / output_id
                        task = asyncio.create_task(metrics_runner.run_and_save_record(output_id, rdir))
                        metrics_background_tasks.append(task)

                    return output_id, result, vr.passed, vr

                except (FileNotFoundError, json.JSONDecodeError) as exc:
                    # Sample-data problems (missing/corrupt artifacts) — mark as invalid
                    # and move on. The orchestrator treats vr=None as "not_finished".
                    logger.warning(f"Skipping {output_id} as invalid: {exc}")
                    return output_id, exc, False, None
                except Exception as exc:
                    # Log loudly but return instead of raising — asyncio.gather does
                    # not use return_exceptions, so a raise here kills result.json
                    # writes for every other record still in flight.
                    logger.error(f"Pipeline error for {output_id}: {exc}", exc_info=True)
                    return output_id, exc, False, None

            pipeline_results = await tqdm.gather(
                *(_run_and_pipeline(output_id_to_record[oid], oid) for oid in pending_output_ids),
                total=len(pending_output_ids),
                desc=f"Attempt {attempt_number}/{max_attempts}",
                file=TqdmNewlineStream(sys.stderr),
                mininterval=0,
                unit="task",
                smoothing=0,  # Smoothing would produce unstable estimates because tasks run in parallel.
                colour="BRIGHT_BLUE",
            )

            # Collect per-record outcomes.
            # vr=None → crash/incomplete (not_finished).
            # vr.passed=False + empty failed_metrics → gate rejected (not_finished).
            # vr.passed=False + non-empty failed_metrics → LLM metrics failed (validation_failed).
            finished_ids: list[str] = []
            not_finished_ids: list[str] = []
            failed_validation_ids: list[str] = []
            validation_results: dict[str, ValidationResult] = {}
            result_map: dict[str, ConversationResult] = {}

            for output_id, _result, passed, vr in pipeline_results:
                if isinstance(_result, ConversationResult):
                    result_map[output_id] = _result
                if vr is None or (not vr.passed and not vr.failed_metrics):
                    not_finished_ids.append(output_id)
                else:
                    finished_ids.append(output_id)
                    if not passed:
                        failed_validation_ids.append(output_id)
                        validation_results[output_id] = vr

            if not_finished_ids:
                logger.info(
                    f"{len(not_finished_ids)} tasks did not finish properly (attempt {attempt_number}/{max_attempts})"
                )

            failed_this_attempt = not_finished_ids + failed_validation_ids

            for oid in not_finished_ids:
                # Distinguish time limit exceeded from other not_finished reasons
                cr = result_map.get(oid)
                hit_time_limit = cr is not None and cr.conversation_ended_reason == "time_limit_exceeded"
                reason = "time_limit_exceeded" if hit_time_limit else "not_finished"
                if hit_time_limit:
                    time_limit_attempt_counts[oid] = time_limit_attempt_counts.get(oid, 0) + 1
                    # Eagerly run LLM validation (skip gate) on every time-limit attempt
                    # and cache the result for later lookup.
                    vr = await pipeline_validation_runner.validate_one(oid, skip_gate=True)
                    time_limit_validation_cache.setdefault(oid, {})[attempt_number] = vr
                rerun_history.setdefault(oid, []).append(
                    {
                        "attempt": attempt_number,
                        "reason": reason,
                    }
                )
            for oid in failed_validation_ids:
                vr = validation_results[oid]
                entry: dict = {
                    "attempt": attempt_number,
                    "reason": "validation_failed",
                    "failed_metrics": vr.failed_metrics,
                    "scores": vr.scores,
                }
                if vr.details:
                    failure_details = {name: vr.details[name] for name in vr.failed_metrics if name in vr.details}
                    if failure_details:
                        entry["failure_details"] = failure_details
                rerun_history.setdefault(oid, []).append(entry)

            # Check for time-limit-accepted records: records that have hit the time limit
            # max_time_limit_attempts times get evaluated with gate bypass.
            # The current attempt was already validated eagerly above; if it passes,
            # accept immediately. Otherwise, scan cached results from previous
            # time-limit attempts and restore the archived directory if one passes.
            newly_time_limit_accepted: list[str] = []
            for oid in list(failed_this_attempt):
                if time_limit_attempt_counts.get(oid, 0) < max_time_limit_attempts:
                    continue

                cached = time_limit_validation_cache.get(oid, {})
                # Check current attempt first
                current_vr = cached.get(attempt_number)
                if current_vr and current_vr.passed:
                    logger.info(
                        f"Record {oid} hit time limit {time_limit_attempt_counts[oid]} times, "
                        f"current attempt passed LLM validation - accepting"
                    )
                    self._accept_time_limit_record(
                        oid,
                        failed_this_attempt,
                        finished_ids,
                        newly_time_limit_accepted,
                        metrics_runner,
                        metrics_background_tasks,
                    )
                    continue

                # Scan previous time-limit attempts for one that passed
                accepted_from_archive = False
                for prev_attempt, prev_vr in cached.items():
                    if prev_attempt == attempt_number:
                        continue
                    if prev_vr.passed:
                        logger.info(
                            f"Record {oid} hit time limit {time_limit_attempt_counts[oid]} times, "
                            f"restoring attempt {prev_attempt} which passed LLM validation"
                        )
                        # Restore the archived attempt directory
                        archive_dir = self.output_dir / "records" / f"{oid}_failed_attempt_{prev_attempt}"
                        record_dir = self.output_dir / "records" / oid
                        if archive_dir.exists():
                            # Move current attempt out of the way, restore the passing one
                            if record_dir.exists():
                                shutil.move(
                                    str(record_dir),
                                    str(self.output_dir / "records" / f"{oid}_failed_attempt_{attempt_number}"),
                                )
                            shutil.move(str(archive_dir), str(record_dir))
                            self._accept_time_limit_record(
                                oid,
                                failed_this_attempt,
                                finished_ids,
                                newly_time_limit_accepted,
                                metrics_runner,
                                metrics_background_tasks,
                            )
                            accepted_from_archive = True
                            break
                        else:
                            logger.warning(
                                f"Record {oid}: archive dir for attempt {prev_attempt} not found at "
                                f"{archive_dir} - cannot restore, skipping"
                            )

                if not accepted_from_archive:
                    logger.info(
                        f"Record {oid} hit time limit {time_limit_attempt_counts[oid]} times, "
                        f"no attempt passed LLM validation - staying pending"
                    )

            if newly_time_limit_accepted:
                time_limit_accepted_ids.update(newly_time_limit_accepted)
                logger.info(f"{len(newly_time_limit_accepted)} time-limit records accepted via gate bypass")

            pending_output_ids = failed_this_attempt

            if not pending_output_ids:
                logger.info("All tasks passed validation!")
                break
            elif attempt_number < max_attempts:
                logger.info(f"Archiving {len(pending_output_ids)} failed tasks for rerun...")
                for output_id in pending_output_ids:
                    self._archive_failed_attempt(output_id, attempt_number)
            else:
                logger.warning(f"{len(pending_output_ids)} tasks still failing after {max_attempts} attempts")

        # STEP 6: Compute final success/failure sets
        final_failed_ids = set(pending_output_ids)
        successful_ids = set(all_output_ids) - final_failed_ids

        # Categorize failures
        not_finished_count = 0
        validation_failed_count = 0
        for oid in final_failed_ids:
            record_dir = self.output_dir / "records" / oid
            if not check_conversation_finished(record_dir):
                not_finished_count += 1
            else:
                validation_failed_count += 1

        # Archive the final failing attempts so the directory layout reflects the
        # failure (downstream tools key off the `_failed_attempt_` suffix).
        for oid in final_failed_ids:
            self._archive_failed_attempt(oid, attempt_number)

        # STEP 7: Await background metrics, then run final aggregation pass.
        # Background tasks already wrote metrics.json for records validated during the loop.
        # IMPORTANT: Must await all background tasks BEFORE mutating metrics_runner.record_ids
        # below — running tasks read record_ids to filter which records to process.
        # The final run() skips already-computed records and only does summary aggregation.
        if metrics_background_tasks:
            logger.info(f"Waiting for {len(metrics_background_tasks)} background metrics tasks...")
            bg_results = await asyncio.gather(*metrics_background_tasks, return_exceptions=True)
            for result in bg_results:
                if isinstance(result, Exception):
                    logger.error(f"Background metrics task failed: {result}", exc_info=True)

        if self.config.metrics and successful_ids:
            logger.info(f"Running final metrics pass on {len(successful_ids)} successful tasks...")
            if metrics_runner is not None:
                # Reuse the pre-created runner; set record filter to successful-only.
                metrics_runner.record_ids = successful_ids
            else:
                # Fallback: create runner now (e.g. if pre-creation failed earlier).
                successful_records = list(
                    {id(output_id_to_record[oid]): output_id_to_record[oid] for oid in successful_ids}.values()
                )
                metrics_runner = MetricsRunner(
                    run_dir=self.output_dir,
                    dataset=successful_records,
                    metric_names=self.config.metrics,
                    metric_configs=self._metric_configs,
                    record_ids=list(successful_ids),
                    num_draws=self.config.num_trials,
                    force_rerun=self.config.force_rerun_metrics,
                )
            await metrics_runner.run()
        elif self.config.metrics and not successful_ids:
            logger.info("Skipping metrics: no records passed validation")

        # STEP 8: Generate final summary
        ended_at = datetime.now()
        total_tasks = len(all_output_ids)
        successful_count = len(successful_ids)
        failed_count = len(final_failed_ids)

        # Build final_failures from the last rerun_history entry for each failed record
        final_failures: dict[str, dict] = {}
        for oid in final_failed_ids:
            if oid in rerun_history and rerun_history[oid]:
                final_failures[oid] = rerun_history[oid][-1]
            else:
                # Failed on initial validation (no rerun history)
                record_dir = self.output_dir / "records" / oid
                if not check_conversation_finished(record_dir):
                    final_failures[oid] = {"reason": "not_finished"}
                else:
                    final_failures[oid] = {"reason": "validation_failed"}

        llm_generic_error_record_ids = find_records_with_llm_generic_error(self.output_dir, successful_ids)
        eval_summary_path = self.output_dir / "evaluation_summary.json"
        with open(eval_summary_path, "w") as f:
            json.dump(
                {
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": (ended_at - started_at).total_seconds(),
                    "simulation": {
                        "total_records": total_tasks,
                        "successful_records": successful_count,
                        "failed_records": failed_count,
                        "not_finished_count": not_finished_count,
                        "validation_failed_count": validation_failed_count,
                        "records_with_llm_generic_error": len(llm_generic_error_record_ids),
                        "llm_generic_error_record_ids": llm_generic_error_record_ids,
                        "success_rate": successful_count / total_tasks if total_tasks > 0 else 0.0,
                        "failure_rate": failed_count / total_tasks if total_tasks > 0 else 0.0,
                        "total_attempts": attempt_number,
                        "failed_record_ids": sorted(final_failed_ids),
                        "successful_record_ids": sorted(successful_ids),
                        "time_limit_accepted_record_ids": sorted(time_limit_accepted_ids),
                    },
                    "rerun_history": rerun_history,
                    "final_failures": final_failures,
                },
                f,
                indent=2,
            )

        # Save CSV with only successful records
        successful_results: list[tuple[str, ConversationResult]] = []
        for output_id in successful_ids:
            result_path = self.output_dir / "records" / output_id / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                    successful_results.append((output_id, ConversationResult(**result_data)))

        self._save_results_csv(successful_results, list(final_failed_ids))

        logger.info("=" * 60)
        logger.info("Benchmark complete:")
        if total_tasks > 0:
            logger.info(f"  Success: {successful_count}/{total_tasks} ({successful_count / total_tasks * 100:.1f}%)")
            logger.info(f"  Failed: {failed_count}/{total_tasks} ({failed_count / total_tasks * 100:.1f}%)")
            if not_finished_count > 0:
                logger.info(f"    Not finished: {not_finished_count}")
            if validation_failed_count > 0:
                logger.info(f"    Validation failed: {validation_failed_count}")
        else:
            logger.info("  No records processed")
        logger.info(f"  Total attempts used: {attempt_number}")
        logger.info("=" * 60)

        return RunResult(
            run_id=self.config.run_id,
            total_records=total_tasks,
            successful_records=successful_count,
            failed_records=failed_count,
            duration_seconds=(ended_at - started_at).total_seconds(),
        )

    async def _run_conversation(
        self, record: EvaluationRecord, output_id: str
    ) -> tuple[ConversationResult, asyncio.Task | None]:
        """Run a single conversation.

        Args:
            record: Evaluation record to run (never modified, record.id used for scenario DB)
            output_id: Directory name for output (may include _trial_N suffix)

        Returns:
            Tuple of (ConversationResult, deferred_audio_task). The audio task writes WAV
            files in a thread pool and should be awaited before accessing audio outputs.
        """
        # Acquire port
        port = await self.port_pool.acquire()

        # Use scenario base path from config
        scenario_base_path = str(self.config.tool_mocks_path)

        try:
            # Create worker with output_id for the directory name.
            # The record object is untouched — worker uses record.id for scenario DB lookup.
            worker = ConversationWorker(
                config=self.config,
                record=record,
                agent=self.agent,
                agent_config_path=str(self.config.agent_config_path),
                scenario_base_path=scenario_base_path,
                output_dir=self.output_dir / "records" / output_id,
                port=port,
                output_id=output_id,
            )

            # Run conversation
            result = await worker.run()
            return result, worker.deferred_audio_task
        finally:
            # Always release port
            await self.port_pool.release(port)

    async def _run_targeted(
        self, tasks: list[tuple[EvaluationRecord, str]]
    ) -> dict[str, ConversationResult | Exception]:
        """Run specific (record, output_id) pairs — single attempt each, no retry.

        Unlike run(), this does NOT expand records into trials
        and does NOT retry on failure. The caller (_run_with_validation) owns
        all retry logic via its flat loop.

        Args:
            tasks: List of (record, output_id) pairs to run.

        Returns:
            Dict mapping output_id -> ConversationResult or Exception.
        """
        if not tasks:
            return {}

        # Ensure output directory and port pool are ready
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "records").mkdir(exist_ok=True)
        await self.port_pool.initialize()

        semaphore = asyncio.Semaphore(self.config.max_concurrent_conversations)

        async def run_with_semaphore(record: EvaluationRecord, output_id: str) -> ConversationResult:
            async with semaphore:
                result, audio_task = await self._run_conversation(record, output_id)
            # Semaphore released; await audio task outside it so the slot is free sooner.
            if audio_task is not None:
                try:
                    await audio_task
                except Exception as e:
                    logger.warning(f"Audio save failed for {output_id}: {e}")
            return result

        coros = [run_with_semaphore(record, oid) for record, oid in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        output: dict[str, ConversationResult | Exception] = {}
        for (_record, output_id), result in zip(tasks, results):
            output[output_id] = result

            # Save result.json for successful ConversationResults
            if isinstance(result, ConversationResult):
                result_path = self.output_dir / "records" / output_id / "result.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(result.model_dump_json(indent=2))

        return output

    async def validate_existing(
        self, run_dir: Path, records: list[EvaluationRecord], force_revalidation: bool = False
    ) -> RunResult:
        """Validate existing output and rerun failures.

        Used for Modes 5/6: point at existing run output, validate it,
        and rerun any records that fail validation. Supports multi-trial
        directories ({record_id}/trial_{N}).

        Args:
            run_dir: Path to existing run directory
            records: Full list of evaluation records
            force_revalidation: If True, re-validate all records even if they
                previously passed. If False (default), skip records listed in
                evaluation_summary.json's successful_record_ids.

        Returns:
            RunResult with final counts and duration
        """
        logger.info(f"Validating existing output in {run_dir}")
        started_at = datetime.now()

        # Use the existing run directory
        self.output_dir = run_dir

        # Apply record filtering
        filtered_records = self._filter_records(records)

        # Build output_id list from config (matches _run_with_validation logic)
        records_dir = self.output_dir / "records"
        num_trials = self.config.num_trials
        output_id_to_record: dict[str, EvaluationRecord] = {}

        for record in filtered_records:
            if num_trials > 1:
                for trial_idx in range(num_trials):
                    oid = f"{record.id}/trial_{trial_idx}"
                    output_id_to_record[oid] = record
            else:
                output_id_to_record[record.id] = record

        all_output_ids = list(output_id_to_record.keys())

        if not all_output_ids:
            logger.warning("No record directories found in existing output")
            ended_at = datetime.now()
            return RunResult(
                run_id=self.config.run_id,
                total_records=0,
                successful_records=0,
                failed_records=0,
                duration_seconds=(ended_at - started_at).total_seconds(),
            )

        # Load previously successful record IDs from evaluation_summary.json
        already_passed_ids: set[str] = set()
        if not force_revalidation:
            eval_summary_path = self.output_dir / "evaluation_summary.json"
            if eval_summary_path.exists():
                with open(eval_summary_path) as f:
                    prev_summary = json.load(f)
                # Support both new format (nested under "simulation") and legacy flat format
                sim = prev_summary.get("simulation", prev_summary)
                prev_successful = set(sim.get("successful_record_ids", []))
                # Only skip IDs that are still present in the current output
                already_passed_ids = prev_successful & set(all_output_ids)
                if already_passed_ids:
                    logger.info(
                        f"Skipping validation for {len(already_passed_ids)} already-passed records "
                        f"(from evaluation_summary.json)"
                    )

        needs_validation_ids = [oid for oid in all_output_ids if oid not in already_passed_ids]

        # STEP 1: Full validation (including conversation_valid_end)
        validation_results: dict = {}
        failed_ids: list[str] = []
        if needs_validation_ids:
            logger.info(f"Running full validation on {len(needs_validation_ids)} existing outputs...")
            validation_runner = ValidationRunner(
                run_dir=self.output_dir,
                dataset=filtered_records,
                thresholds=self.config.validation_thresholds,
                metric_configs=self._metric_configs,
                output_ids=needs_validation_ids,
            )
            validation_results = await validation_runner.run_validation()

            # STEP 2: Identify failures (only among records that needed validation)
            for output_id in needs_validation_ids:
                vr = validation_results.get(output_id)
                if not vr or not vr.passed:
                    failed_ids.append(output_id)
        else:
            logger.info("All records already passed validation — nothing to validate")

        newly_passed_count = len(needs_validation_ids) - len(failed_ids)
        total_passed = newly_passed_count + len(already_passed_ids)
        logger.info(f"Validation results: {total_passed} passed, {len(failed_ids)} failed")

        # STEP 3: Rerun failures using the same flat loop as _run_with_validation
        max_attempts = self.config.max_rerun_attempts
        rerun_history: dict[str, list[dict]] = {}
        pending_ids = list(failed_ids)

        for attempt_number in range(1, max_attempts + 1):
            if not pending_ids:
                break

            # Archive failed outputs
            for output_id in pending_ids:
                self._archive_failed_attempt(output_id, attempt_number)

            tasks = [(output_id_to_record[oid], oid) for oid in pending_ids]
            run_results = await self._run_targeted(tasks)

            still_not_finished: list[str] = []
            to_validate_ids: list[str] = []
            for output_id in pending_ids:
                result = run_results.get(output_id)
                if isinstance(result, Exception) or (isinstance(result, ConversationResult) and not result.completed):
                    still_not_finished.append(output_id)
                else:
                    to_validate_ids.append(output_id)

            failed_validation: list[str] = []
            new_results: dict = {}
            if to_validate_ids:
                to_validate_records = list(
                    {id(output_id_to_record[oid]): output_id_to_record[oid] for oid in to_validate_ids}.values()
                )
                vr_runner = ValidationRunner(
                    run_dir=self.output_dir,
                    dataset=to_validate_records,
                    thresholds=self.config.validation_thresholds,
                    metric_configs=self._metric_configs,
                    output_ids=to_validate_ids,
                )
                new_results = await vr_runner.run_validation()
                for output_id in to_validate_ids:
                    vr = new_results.get(output_id)
                    if vr is None or (not vr.passed and not vr.failed_metrics):
                        still_not_finished.append(output_id)
                    elif not vr.passed:
                        failed_validation.append(output_id)

            pending_ids = still_not_finished + failed_validation
            for oid in still_not_finished:
                rerun_history.setdefault(oid, []).append(
                    {
                        "attempt": attempt_number,
                        "reason": "not_finished",
                    }
                )
            for oid in failed_validation:
                vr = new_results[oid]
                entry: dict = {
                    "attempt": attempt_number,
                    "reason": "validation_failed",
                    "failed_metrics": vr.failed_metrics,
                    "scores": vr.scores,
                }
                if vr.details:
                    failure_details = {name: vr.details[name] for name in vr.failed_metrics if name in vr.details}
                    if failure_details:
                        entry["failure_details"] = failure_details
                rerun_history.setdefault(oid, []).append(entry)

            if not pending_ids:
                logger.info("All rerun records now pass validation!")
                break

        if pending_ids:
            logger.warning(f"{len(pending_ids)} tasks still failing after {max_attempts} attempts")

        # STEP 4: Optionally run metrics
        final_failed_ids = set(pending_ids)
        successful_ids = (set(all_output_ids) - final_failed_ids) | already_passed_ids

        metrics_result: MetricsRunResult | None = None
        if self.config.metrics and successful_ids:
            logger.info(f"Running metrics on {len(successful_ids)} successful records...")
            successful_records = list(
                {id(output_id_to_record[oid]): output_id_to_record[oid] for oid in successful_ids}.values()
            )
            metrics_runner = MetricsRunner(
                run_dir=self.output_dir,
                dataset=successful_records,
                metric_names=self.config.metrics,
                metric_configs=self._metric_configs,
                record_ids=list(successful_ids),
                num_draws=self.config.num_trials,
                force_rerun=self.config.force_rerun_metrics,
            )
            metrics_result = await metrics_runner.run()
        elif self.config.metrics and not successful_ids:
            logger.info("Skipping metrics: no records passed validation")

        # STEP 5: Generate summary
        ended_at = datetime.now()
        total = len(all_output_ids)

        # Categorize final failures
        not_finished_count = sum(1 for oid in final_failed_ids if not check_conversation_finished(records_dir / oid))
        validation_failed_count = len(final_failed_ids) - not_finished_count

        # Build final_failures from the last rerun_history entry for each failed record
        final_failures: dict[str, dict] = {}
        for oid in final_failed_ids:
            if oid in rerun_history and rerun_history[oid]:
                final_failures[oid] = rerun_history[oid][-1]
            else:
                # Failed on initial validation (no rerun history)
                if not check_conversation_finished(records_dir / oid):
                    final_failures[oid] = {"reason": "not_finished"}
                else:
                    vr = validation_results.get(oid)
                    entry: dict = {
                        "reason": "validation_failed",
                        "failed_metrics": vr.failed_metrics if vr else [],
                        "scores": vr.scores if vr else {},
                    }
                    if vr and vr.details:
                        failure_details = {}
                        for metric_name in vr.failed_metrics:
                            if metric_name in vr.details:
                                failure_details[metric_name] = vr.details[metric_name]
                        if failure_details:
                            entry["failure_details"] = failure_details
                    final_failures[oid] = entry

        # Archive the final failing attempts so the directory layout reflects the
        # failure (downstream tools key off the `_failed_attempt_` suffix).
        for oid in final_failed_ids:
            self._archive_failed_attempt(oid, max_attempts)

        # Build evaluation summary with separate simulation and metrics sections
        llm_generic_error_record_ids = find_records_with_llm_generic_error(self.output_dir, successful_ids)
        eval_summary: dict[str, Any] = {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "simulation": {
                "total_records": total,
                "successful_records": len(successful_ids),
                "failed_records": len(final_failed_ids),
                "not_finished_count": not_finished_count,
                "validation_failed_count": validation_failed_count,
                "records_with_llm_generic_error": len(llm_generic_error_record_ids),
                "llm_generic_error_record_ids": llm_generic_error_record_ids,
                "total_rerun_attempts": max(len(v) for v in rerun_history.values()) if rerun_history else 0,
                "failed_record_ids": sorted(final_failed_ids),
                "successful_record_ids": sorted(successful_ids),
            },
            "rerun_history": rerun_history,
            "final_failures": final_failures,
        }

        if metrics_result is not None:
            eval_summary["metrics"] = {
                "records_evaluated": metrics_result.total_records,
                "metrics_computed": self.config.metrics,
                "total_metric_failures": metrics_result.total_metric_failures,
                "metric_failures": {
                    name: sorted(record_ids) for name, record_ids in metrics_result.metric_failures.items()
                },
            }

        eval_summary_path = self.output_dir / "evaluation_summary.json"
        with open(eval_summary_path, "w") as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)

        # Terminal output — clearly separate simulation from metrics
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info("Simulation:")
        logger.info(f"  Successful: {len(successful_ids)}/{total}")
        if final_failed_ids:
            logger.info(f"  Failed: {len(final_failed_ids)}/{total}")

        if self.config.metrics:
            logger.info("Metrics:")
            if metrics_result is None:
                logger.info("  Skipped (no records passed simulation validation)")
            elif metrics_result.has_metric_failures:
                logger.error(
                    f"  {metrics_result.total_metric_failures} metric computation(s) failed "
                    f"across {len(metrics_result.metric_failures)} metric(s): "
                    f"{', '.join(metrics_result.metric_failures.keys())}"
                )
                logger.info("  See METRICS RESULTS above and metrics_summary.json for details.")
            else:
                logger.info(f"  All metrics computed successfully on {metrics_result.total_records} records.")
        logger.info("=" * 60)

        return RunResult(
            run_id=self.config.run_id,
            total_records=total,
            successful_records=len(successful_ids),
            failed_records=len(final_failed_ids),
            duration_seconds=(ended_at - started_at).total_seconds(),
        )

    async def rerun_failed_metrics(
        self,
        run_dir: Path,
        records: list[EvaluationRecord],
        cli_metrics: list[str] | None = None,
    ) -> RunResult:
        """Rerun only previously failed metric computations.

        Reads metric_failures from evaluation_summary.json and reruns only
        the specific failed metrics on the specific failed records. Existing
        successful metric values are preserved and read from disk.

        Args:
            run_dir: Path to existing run directory
            records: Full list of evaluation records
            cli_metrics: CLI metrics

        Returns:
            RunResult with final counts and duration
        """
        logger.info(f"Rerunning failed metrics in {run_dir}")
        started_at = datetime.now()
        self.output_dir = run_dir

        # Read evaluation_summary.json
        eval_summary_path = run_dir / "evaluation_summary.json"
        if not eval_summary_path.exists():
            raise FileNotFoundError(
                f"evaluation_summary.json not found in {run_dir}. "
                "Run metrics first before using --rerun-failed-metrics."
            )

        with open(eval_summary_path) as f:
            eval_summary = json.load(f)

        # Support both old and new schema
        sim = eval_summary.get("simulation", eval_summary)
        successful_ids = sim.get("successful_record_ids", [])

        metrics_section = eval_summary.get("metrics", {})
        metrics_computed = metrics_section.get("metrics_computed", [])

        # Use explicit CLI --metrics if provided, otherwise prefer metrics_computed
        # from evaluation_summary.json (reflects actual metric names from the last run),
        # falling back to config.json metrics as a last resort.
        metric_names = cli_metrics or metrics_computed or self.config.metrics
        if not metric_names:
            raise ValueError(
                "No metrics to run. Specify --metrics or ensure evaluation_summary.json has metrics_computed."
            )

        # Build record_metric_filter by scanning each successful record's metrics.json on
        # disk, rather than trusting the (possibly stale) metric_failures snapshot from
        # the original run. A metric needs rerunning if it is missing, errored, OR has a
        # None score — all three mean it never produced a usable value. Scanning disk also
        # picks up records newly added by reclassification, which the old snapshot missed.
        records_dir = run_dir / "records"
        metric_names_set = set(metric_names)
        record_metric_filter: dict[str, set[str]] = {}
        for record_id in successful_ids:
            metrics_path = records_dir / record_id / "metrics.json"
            existing: dict[str, Any] = {}
            if metrics_path.exists():
                try:
                    existing = json.loads(metrics_path.read_text()).get("metrics", {})
                except Exception as e:
                    logger.warning(f"Could not read {metrics_path}: {e}")
            for name in metric_names_set:
                entry = existing.get(name)
                needs_rerun = (
                    not isinstance(entry, dict)
                    or entry.get("error") is not None
                    or entry.get("score") is None
                )
                if needs_rerun:
                    record_metric_filter.setdefault(record_id, set()).add(name)

        if not record_metric_filter:
            logger.info("No missing/failed/none-score metrics to rerun")
            ended_at = datetime.now()
            return RunResult(
                run_id=self.config.run_id,
                total_records=len(successful_ids),
                successful_records=len(successful_ids),
                failed_records=0,
                duration_seconds=(ended_at - started_at).total_seconds(),
            )

        total_reruns = sum(len(metrics) for metrics in record_metric_filter.values())
        logger.info(
            f"Rerunning {total_reruns} failed metric computation(s) across {len(record_metric_filter)} record(s)"
        )
        per_metric_counts: dict[str, int] = {}
        for metrics in record_metric_filter.values():
            for name in metrics:
                per_metric_counts[name] = per_metric_counts.get(name, 0) + 1
        for name, count in sorted(per_metric_counts.items()):
            logger.info(f"  {name}: {count} record(s)")

        # Create MetricsRunner with all metrics but filter per-record.
        # Records not in record_metric_filter will read existing metrics from disk.
        # Records in the filter will only recompute the failed metrics and merge.
        metrics_runner = MetricsRunner(
            run_dir=run_dir,
            dataset=records,
            metric_names=metric_names,
            metric_configs=self._metric_configs,
            record_ids=successful_ids,
            num_draws=self.config.num_trials,
            record_metric_filter=record_metric_filter,
        )
        metrics_result = await metrics_runner.run()

        # Update evaluation_summary.json with new metrics status
        eval_summary["metrics"] = {
            "records_evaluated": metrics_result.total_records,
            "metrics_computed": metric_names,
            "total_metric_failures": metrics_result.total_metric_failures,
            "metric_failures": {
                name: sorted(record_ids) for name, record_ids in metrics_result.metric_failures.items()
            },
        }

        with open(eval_summary_path, "w") as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)

        # Terminal output
        logger.info("=" * 60)
        logger.info("RERUN FAILED METRICS COMPLETE")
        logger.info("=" * 60)
        if metrics_result.has_metric_failures:
            logger.error(
                f"  {metrics_result.total_metric_failures} metric computation(s) still failing "
                f"across {len(metrics_result.metric_failures)} metric(s)"
            )
            logger.info("  Run with --rerun-failed-metrics again or check error details.")
        else:
            logger.info("  All previously failed metrics now computed successfully!")
        logger.info("=" * 60)

        ended_at = datetime.now()
        return RunResult(
            run_id=self.config.run_id,
            total_records=len(successful_ids),
            successful_records=len(successful_ids),
            failed_records=0,
            duration_seconds=(ended_at - started_at).total_seconds(),
        )

    @staticmethod
    def _errored_validation_metrics(record_dir: Path) -> list[str]:
        """Return validation metrics whose stored value carries an ``error``.

        A non-empty result means the prior judging "did not go smoothly" (e.g. the
        judge hit a transient/config error), which is what makes an archived attempt a
        promotion candidate. An attempt whose validation metrics were all cleanly scored
        — a genuine failure — returns an empty list and is not a candidate.
        """
        metrics_path = record_dir / "metrics.json"
        if not metrics_path.exists():
            return []
        try:
            metrics = json.loads(metrics_path.read_text()).get("metrics", {})
        except Exception as e:
            logger.warning(f"Could not read {metrics_path} to inspect errored metrics: {e}")
            return []
        return [
            name
            for name in ValidationRunner.VALIDATION_METRICS
            if isinstance(metrics.get(name), dict) and metrics[name].get("error") is not None
        ]

    @staticmethod
    def _clear_metrics(record_dir: Path, names: list[str]) -> None:
        """Drop the named metric entries from a record's metrics.json.

        Removed metrics are treated as "missing" by MetricsRunner, forcing a fresh
        re-judge on the next validation pass. Genuine (non-errored) scores are left in
        place so they are reused rather than recomputed.
        """
        metrics_path = record_dir / "metrics.json"
        if not metrics_path.exists() or not names:
            return
        try:
            data = json.loads(metrics_path.read_text())
        except Exception as e:
            logger.warning(f"Could not read {metrics_path} to clear metrics: {e}")
            return
        metrics = data.get("metrics", {})
        removed = [n for n in names if n in metrics]
        if not removed:
            return
        for n in removed:
            del metrics[n]
        metrics_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"{record_dir.name}: cleared errored validation metrics for re-judge: {removed}")

    async def reclassify_run(
        self,
        run_dir: Path,
        records: list[EvaluationRecord],
        cli_metrics: list[str] | None = None,
    ) -> RunResult:
        """Reclassify failed records by re-judging archived attempts that erred.

        Goal: bring each record up to ``num_trials`` successful trials by reusing its own
        already-run simulations. A record that is short of ``num_trials`` draws on the
        pool of *all* its archived failed attempts (across every trial index), not just
        the attempts tied to one slot.

        Candidate rule — the key to not being wasteful and to never "forcing" a pass: an
        archived attempt is a promotion candidate only if its prior judging *did not go
        smoothly*, i.e. a validation metric was stored with an ``error`` (e.g. the judge
        hit a bad API key). Attempts that were cleanly scored as genuine failures are
        left alone. For a candidate, only the errored validation metrics are re-judged;
        cleanly-scored ones are reused.

        Per record, candidates are re-judged serially and the record stops as soon as it
        reaches ``num_trials`` — so a record may inspect many attempts, but never more
        than it needs. Different records are processed concurrently, bounded by the run's
        concurrency limit; the LiteLLM Router further bounds judge API calls.

        Each passing candidate is promoted (moved) into a free ``trial_{N}`` slot. Then
        evaluation_summary.json and results.csv are reconciled, the full metric suite is
        computed on promoted slots (reusing existing metrics), and every promotion is
        recorded in reclassification_log.json.

        Args:
            run_dir: Path to existing run directory.
            records: Full list of evaluation records.
            cli_metrics: Metrics to compute on promoted records (defaults to config).

        Returns:
            RunResult with post-reclassification counts and duration.
        """
        logger.info(f"Reclassifying failed records in {run_dir}")
        started_at = datetime.now()
        self.output_dir = run_dir
        records_dir = run_dir / "records"
        num_trials = self.config.num_trials

        eval_summary_path = run_dir / "evaluation_summary.json"
        if not eval_summary_path.exists():
            raise FileNotFoundError(
                f"evaluation_summary.json not found in {run_dir}. Nothing to reclassify."
            )

        with open(eval_summary_path) as f:
            eval_summary = json.load(f)

        sim = eval_summary.get("simulation", eval_summary)
        failed_ids: list[str] = list(sim.get("failed_record_ids", []))
        successful_ids: list[str] = list(sim.get("successful_record_ids", []))

        def _no_op_result() -> RunResult:
            ended = datetime.now()
            return RunResult(
                run_id=self.config.run_id,
                total_records=len(successful_ids) + len(failed_ids),
                successful_records=len(successful_ids),
                failed_records=len(failed_ids),
                duration_seconds=(ended - started_at).total_seconds(),
            )

        if not failed_ids:
            logger.info("No failed records in evaluation_summary.json — nothing to reclassify")
            return _no_op_result()

        # ── Group by base record; compute how many trials each still needs ──
        record_id_filter = set(self.config.record_ids) if self.config.record_ids else None

        # Records that appear in failed_ids are the ones potentially short of num_trials.
        needy_records: set[str] = set()
        for oid in failed_ids:
            base, _ = parse_trial_record_id(oid)
            if record_id_filter is None or base in record_id_filter:
                needy_records.add(base)

        if not needy_records:
            logger.info("No failed records match the requested --record-ids — nothing to reclassify")
            return _no_op_result()

        all_records = list({id(r): r for r in records}.values())
        validation_runner = ValidationRunner(
            run_dir=run_dir,
            dataset=all_records,
            thresholds=self.config.validation_thresholds,
            metric_configs=self._metric_configs,
        )

        # Different records run concurrently; candidates within a record run serially so
        # we stop the moment num_trials is reached (no wasted judge compute).
        semaphore = asyncio.Semaphore(self.config.max_concurrent_conversations)
        successful_set = set(successful_ids)

        def _slot_output_id(base: str, i: int) -> str:
            return f"{base}/trial_{i}" if num_trials > 1 else base

        def _free_slots(base: str) -> list[int]:
            """Trial indices NOT yet backed by a *successful* trial.

            A slot is satisfied only if its output id is in the successful set from the
            summary — NOT merely because a canonical dir exists on disk. A canonical dir
            can be a leftover from an interrupted prior promotion (moved into place but
            never recorded); such a slot is still free and its dir is re-judged below.
            """
            return [i for i in range(num_trials) if _slot_output_id(base, i) not in successful_set]

        def _candidate_pool(base: str, free: list[int]) -> list[Path]:
            """Re-judge candidates for a record, newest attempt first.

            Includes archived ``_failed_attempt_`` dirs AND any leftover/failed canonical
            ``trial_{i}`` dir for a free slot (from an interrupted prior run).
            """
            base_dir = records_dir / base
            if num_trials > 1:
                archived = base_dir.glob("trial_*_failed_attempt_*") if base_dir.exists() else []
            else:
                archived = records_dir.glob(f"{base}_failed_attempt_*")
            cands = [d for d in archived if d.is_dir() and ATTEMPT_SUFFIX_PATTERN.search(d.name)]
            cands.sort(key=lambda d: int(ATTEMPT_SUFFIX_PATTERN.search(d.name).group(2)), reverse=True)
            # Prepend leftover canonical dirs for free slots so an already-in-place
            # attempt is tried first (and, if it passes, needs no move).
            leftovers = [
                records_dir / _slot_output_id(base, i)
                for i in free
                if (records_dir / _slot_output_id(base, i)).exists()
            ]
            return leftovers + cands

        async def _reclassify_record(base: str) -> list[dict]:
            free = _free_slots(base)
            if not free:
                logger.info(f"{base}: already has {num_trials} successful trial(s) — skipping")
                return []
            needed = len(free)
            candidates = _candidate_pool(base, free)
            if not candidates:
                logger.info(f"{base}: no attempts to re-judge — leaving failed")
                return []

            promotions: list[dict] = []
            async with semaphore:
                for cand in candidates:
                    if not free:
                        break
                    # Candidate rule: only attempts whose prior judging erred. Cleanly
                    # scored genuine failures are skipped — never re-judged or forced.
                    errored = self._errored_validation_metrics(cand)
                    if not errored:
                        continue
                    self._clear_metrics(cand, errored)
                    cand_output_id = str(cand.relative_to(records_dir))
                    vr = await validation_runner.validate_one(cand_output_id)
                    if not vr.passed:
                        logger.info(f"{base}: re-judged '{cand.name}' still fails validation")
                        continue

                    slot_idx = free.pop(0)
                    target = records_dir / _slot_output_id(base, slot_idx)
                    if cand.resolve() != target.resolve():
                        # A leftover/failed canonical dir may be occupying the target slot;
                        # archive it aside before moving the winning candidate into place.
                        if target.exists():
                            self._archive_failed_attempt(_slot_output_id(base, slot_idx), 1)
                        shutil.move(str(cand), str(target))
                    logger.info(
                        f"{base}: promoted '{cand.name}' into 'trial_{slot_idx}' — now passes "
                        f"validation (scores={vr.scores})"
                    )
                    promotions.append(
                        {
                            "output_id": _slot_output_id(base, slot_idx),
                            "record_id": base,
                            "promoted_from": cand.name,
                            "reclassified_at": started_at.isoformat(),
                        }
                    )
            filled = len(promotions)
            if filled < needed:
                logger.info(
                    f"{base}: filled {filled}/{needed} needed trial(s) — "
                    f"{needed - filled} slot(s) remain failed"
                )
            return promotions

        record_results = await asyncio.gather(*(_reclassify_record(b) for b in sorted(needy_records)))
        reclassification_log = [p for record_promos in record_results for p in record_promos]
        promoted = [p["output_id"] for p in reclassification_log]

        if not promoted:
            logger.info("No candidate attempts passed re-judging — nothing promoted")
            return _no_op_result()

        # ── Reconcile evaluation_summary.json ──
        promoted_set = set(promoted)
        new_successful = sorted(set(successful_ids) | promoted_set)
        new_failed = sorted(set(sim.get("failed_record_ids", [])) - promoted_set)
        total = len(new_successful) + len(new_failed)

        sim["successful_record_ids"] = new_successful
        sim["failed_record_ids"] = new_failed
        sim["successful_records"] = len(new_successful)
        sim["failed_records"] = len(new_failed)
        sim["success_rate"] = len(new_successful) / total if total > 0 else 0.0
        sim["failure_rate"] = len(new_failed) / total if total > 0 else 0.0

        final_failures = eval_summary.get("final_failures", {})
        for oid in promoted_set:
            final_failures.pop(oid, None)
        eval_summary["final_failures"] = final_failures
        eval_summary["reclassification_history"] = (
            eval_summary.get("reclassification_history", []) + reclassification_log
        )

        with open(eval_summary_path, "w") as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)

        # Audit trail of exactly what moved.
        with open(run_dir / "reclassification_log.json", "w") as f:
            json.dump(reclassification_log, f, indent=2, ensure_ascii=False)

        # ── Rebuild results.csv from the reconciled successful set ──
        successful_results: list[tuple[str, ConversationResult]] = []
        for output_id in new_successful:
            result_path = records_dir / output_id / "result.json"
            if result_path.exists():
                with open(result_path) as rf:
                    successful_results.append((output_id, ConversationResult(**json.load(rf))))
        self._save_results_csv(successful_results, new_failed)

        logger.info(f"Promoted {len(promoted)} slot(s): {', '.join(promoted)}")

        # ── Compute full metrics on promoted slots (reuses existing validation metrics) ──
        metric_names = cli_metrics or self.config.metrics
        if metric_names:
            logger.info(f"Computing metrics on {len(promoted)} promoted slot(s)...")
            metrics_runner = MetricsRunner(
                run_dir=run_dir,
                dataset=records,
                metric_names=metric_names,
                metric_configs=self._metric_configs,
                record_ids=promoted,
                num_draws=self.config.num_trials,
            )
            await metrics_runner.run()
        else:
            logger.info("No metrics configured — skipping metric computation on promoted slots")

        logger.info("=" * 60)
        logger.info("RECLASSIFICATION COMPLETE")
        logger.info(f"  Promoted: {len(promoted)}")
        logger.info(f"  Remaining failed: {len(new_failed)}")
        logger.info("=" * 60)

        ended_at = datetime.now()
        return RunResult(
            run_id=self.config.run_id,
            total_records=total,
            successful_records=len(new_successful),
            failed_records=len(new_failed),
            duration_seconds=(ended_at - started_at).total_seconds(),
        )

    @classmethod
    def from_existing_run(cls, run_dir: Path) -> "BenchmarkRunner":
        """Create a BenchmarkRunner from an existing run's config.json.

        Args:
            run_dir: Path to existing run directory containing config.json

        Returns:
            Configured BenchmarkRunner with output_dir pointing to the existing run
        """
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {run_dir}")

        # Load the saved config without reading from env vars or .env file.
        # This prevents conflicts when the current environment has a different pipeline
        # mode set (e.g. EVA_MODEL__LLM in env but the saved run used S2S).
        class _StoredRunConfig(RunConfig):
            @classmethod
            def settings_customise_sources(cls, settings_cls, init_settings, **kwargs):
                return (init_settings,)

        config_data = json.loads(config_path.read_text())
        # Backwards compat: remap any legacy metric names saved in an older config.json.
        if isinstance(config_data.get("metrics"), list):
            config_data["metrics"] = rename_metric_list(config_data["metrics"])
        if isinstance(config_data.get("validation_thresholds"), dict):
            config_data["validation_thresholds"] = rename_metric_keys(config_data["validation_thresholds"])

        config = _StoredRunConfig(**config_data)
        runner = cls(config)
        runner.output_dir = run_dir  # Use existing output dir, don't create new
        return runner

    def _accept_time_limit_record(
        self,
        oid: str,
        failed_this_attempt: list[str],
        finished_ids: list[str],
        newly_time_limit_accepted: list[str],
        metrics_runner: MetricsRunner | None,
        metrics_background_tasks: list[asyncio.Task],
    ) -> None:
        """Accept a time-limit record by updating result.json and scheduling metrics."""
        failed_this_attempt.remove(oid)
        finished_ids.append(oid)
        newly_time_limit_accepted.append(oid)
        # Update result.json with time_limit_accepted flag
        result_path = self.output_dir / "records" / oid / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            result_data["time_limit_accepted"] = True
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)
        # Fire metrics if runner available
        if metrics_runner is not None:
            rdir = self.output_dir / "records" / oid
            task = asyncio.create_task(metrics_runner.run_and_save_record(oid, rdir))
            metrics_background_tasks.append(task)

    def _archive_failed_attempt(self, record_id: str, attempt_number: int) -> None:
        """Archive a failed attempt before rerunning.

        Args:
            record_id: ID of the record to archive
            attempt_number: Attempt number being archived
        """
        record_dir = self.output_dir / "records" / record_id
        if not record_dir.exists():
            return

        # Find the next available attempt number (previous runs may have left archives)
        archive_dir = self.output_dir / "records" / f"{record_id}_failed_attempt_{attempt_number}"
        while archive_dir.exists():
            attempt_number += 1
            archive_dir = self.output_dir / "records" / f"{record_id}_failed_attempt_{attempt_number}"

        shutil.move(str(record_dir), str(archive_dir))
        logger.debug(f"Archived {record_id} attempt {attempt_number} to {archive_dir}")

    def _save_results_csv(
        self,
        successful: list[tuple[str, ConversationResult]],
        failed_ids: list[str],
    ) -> None:
        """Save results as CSV for easy analysis."""
        csv_path = self.output_dir / "results.csv"

        with open(csv_path, "w") as f:
            # Header
            f.write("record_id,completed,duration_seconds,num_turns,num_tool_calls,ended_reason,error\n")

            # Successful records
            f.writelines(
                f"{output_id},true,{result.duration_seconds:.2f},"
                f"{result.num_turns},{result.num_tool_calls},"
                f"{result.conversation_ended_reason or ''},\n"
                for output_id, result in successful
            )

            # Failed records
            f.writelines(f"{record_id},false,0,0,0,error,failed\n" for record_id in failed_ids)

    @classmethod
    def from_config_file(cls, config_path: Path | str) -> "BenchmarkRunner":
        """Create a BenchmarkRunner from a config file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configured BenchmarkRunner
        """
        config = RunConfig.from_yaml(config_path)
        return cls(config)
