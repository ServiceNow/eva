"""Benchmark runner - main orchestrator for running voice agent benchmarks."""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from eva.metrics.legacy_aliases import rename_metric_keys, rename_metric_list
from eva.metrics.runner import MetricsRunner, MetricsRunResult
from eva.models.agents import AgentConfig
from eva.models.config import PipelineConfig, RunConfig
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, RunResult
from eva.orchestrator.port_pool import PortPool
from eva.orchestrator.validation_runner import ValidationRunner
from eva.orchestrator.worker import ConversationWorker
from eva.utils.conversation_checks import check_conversation_finished, find_records_with_llm_generic_error
from eva.utils.logging import get_logger
from eva.utils.provenance import capture_provenance, resolve_tool_module_file

logger = get_logger(__name__)


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

        # Results tracking
        self._results: list[ConversationResult] = []
        self._failed_record_ids: list[str] = []

    def _load_agent_config(self) -> AgentConfig:
        """Load single agent configuration."""
        return AgentConfig.from_yaml(self.config.agent_config_path)

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
        if isinstance(self.config.model, PipelineConfig):
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
        config_path.write_text(json.dumps(config_data, indent=2))

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
        started_at = datetime.now()

        # LOOP: Run and validate, up to max_attempts total
        for attempt_number in range(1, max_attempts + 1):
            if not pending_output_ids:
                break

            logger.info(
                f"\n{'=' * 60}\n"
                f"Attempt {attempt_number}/{max_attempts}: "
                f"Running {len(pending_output_ids)} tasks\n"
                f"{'=' * 60}"
            )

            # STEP 1: Run conversations for pending output_ids only
            tasks = [(output_id_to_record[oid], oid) for oid in pending_output_ids]
            run_results = await self._run_targeted(tasks)

            not_finished_ids: list[str] = []
            to_validate_ids: list[str] = []
            for output_id in pending_output_ids:
                result = run_results.get(output_id)
                if isinstance(result, Exception) or (isinstance(result, ConversationResult) and not result.completed):
                    not_finished_ids.append(output_id)
                else:
                    to_validate_ids.append(output_id)

            validation_results: dict = {}
            failed_validation_ids: list[str] = []
            if to_validate_ids:
                to_validate_records = list(
                    {id(output_id_to_record[oid]): output_id_to_record[oid] for oid in to_validate_ids}.values()
                )
                logger.info(f"Running validation on {len(to_validate_ids)} tasks...")
                validation_runner = ValidationRunner(
                    run_dir=self.output_dir,
                    dataset=to_validate_records,
                    thresholds=self.config.validation_thresholds,
                    output_ids=to_validate_ids,
                )
                validation_results = await validation_runner.run_validation()

                for output_id in to_validate_ids:
                    vr = validation_results.get(output_id)
                    if vr is None or (not vr.passed and not vr.failed_metrics):
                        not_finished_ids.append(output_id)
                    elif not vr.passed:
                        failed_validation_ids.append(output_id)

            if not_finished_ids:
                logger.info(
                    f"{len(not_finished_ids)} tasks did not finish properly (attempt {attempt_number}/{max_attempts})"
                )

            failed_this_attempt = not_finished_ids + failed_validation_ids

            for oid in not_finished_ids:
                rerun_history.setdefault(oid, []).append(
                    {
                        "attempt": attempt_number,
                        "reason": "not_finished",
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

        # STEP 7: Run full metrics on successful records
        if self.config.metrics and successful_ids:
            logger.info(f"Running full metrics suite on {len(successful_ids)} successful tasks...")
            successful_records = list(
                {id(output_id_to_record[oid]): output_id_to_record[oid] for oid in successful_ids}.values()
            )
            metrics_runner = MetricsRunner(
                run_dir=self.output_dir,
                dataset=successful_records,
                metric_names=self.config.metrics,
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

        logger.info(f"\n{'=' * 60}")
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
        logger.info(f"{'=' * 60}\n")

        return RunResult(
            run_id=self.config.run_id,
            total_records=total_tasks,
            successful_records=successful_count,
            failed_records=failed_count,
            duration_seconds=(ended_at - started_at).total_seconds(),
        )

    async def _run_conversation(self, record: EvaluationRecord, output_id: str) -> ConversationResult:
        """Run a single conversation.

        Args:
            record: Evaluation record to run (never modified, record.id used for scenario DB)
            output_id: Directory name for output (may include _trial_N suffix)

        Returns:
            ConversationResult
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
            return await worker.run()
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
                return await self._run_conversation(record, output_id)

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

            logger.info(
                f"\n{'=' * 60}\nRerun attempt {attempt_number}/{max_attempts}: {len(pending_ids)} tasks\n{'=' * 60}"
            )

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
            json.dump(eval_summary, f, indent=2)

        # Terminal output — clearly separate simulation from metrics
        logger.info(f"{'=' * 60}")
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
        metric_failures = metrics_section.get("metric_failures", {})
        metrics_computed = metrics_section.get("metrics_computed", [])

        if not metric_failures:
            logger.info("No metric failures found in evaluation_summary.json — nothing to rerun")
            ended_at = datetime.now()
            return RunResult(
                run_id=self.config.run_id,
                total_records=len(successful_ids),
                successful_records=len(successful_ids),
                failed_records=0,
                duration_seconds=(ended_at - started_at).total_seconds(),
            )

        # Use explicit CLI --metrics if provided, otherwise prefer metrics_computed
        # from evaluation_summary.json (reflects actual metric names from the last run),
        # falling back to config.json metrics as a last resort.
        metric_names = cli_metrics or metrics_computed or self.config.metrics
        if not metric_names:
            raise ValueError(
                "No metrics to run. Specify --metrics or ensure evaluation_summary.json has metrics_computed."
            )

        # Build record_metric_filter: record_id -> set of metric names to rerun
        successful_set = set(successful_ids)
        record_metric_filter: dict[str, set[str]] = {}
        for metric_name, failed_record_ids in metric_failures.items():
            if metric_name in metric_names:
                for record_id in failed_record_ids:
                    if record_id in successful_set:
                        record_metric_filter.setdefault(record_id, set()).add(metric_name)

        if not record_metric_filter:
            logger.info("No applicable metric failures to rerun")
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
        for metric_name, failed_ids in metric_failures.items():
            if metric_name in metric_names:
                applicable = [rid for rid in failed_ids if rid in record_metric_filter]
                if applicable:
                    logger.info(f"  {metric_name}: {len(applicable)} record(s)")

        # Create MetricsRunner with all metrics but filter per-record.
        # Records not in record_metric_filter will read existing metrics from disk.
        # Records in the filter will only recompute the failed metrics and merge.
        metrics_runner = MetricsRunner(
            run_dir=run_dir,
            dataset=records,
            metric_names=metric_names,
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
            json.dump(eval_summary, f, indent=2)

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
