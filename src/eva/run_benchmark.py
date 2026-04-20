"""Core benchmark orchestration — imported by eva.cli."""

import asyncio
import sys

from dotenv import load_dotenv

from eva.metrics.runner import MetricsRunner
from eva.models.config import PipelineConfig, RunConfig
from eva.models.record import EvaluationRecord
from eva.orchestrator.runner import BenchmarkRunner
from eva.utils import router
from eva.utils.logging import get_logger, setup_logging

load_dotenv()


async def run_benchmark(config: RunConfig) -> int:
    """Run the benchmark end-to-end and return an exit code."""
    # Install custom hook to suppress threading cleanup errors at shutdown
    sys.unraisablehook = _suppress_threading_cleanup_error

    setup_logging(level=config.log_level)
    logger = get_logger(__name__)
    router.init(config.model_list)

    # Check if run_id points to an existing run
    resolved_dir = config.output_dir / config.run_id
    existing_run = (resolved_dir / "config.json").exists()

    if existing_run and config.aggregate_only:
        # ── Aggregate-only: recompute EVA composites from existing metrics ──
        await MetricsRunner.run_aggregate_only(resolved_dir, num_draws=config.num_trials)
        logger.info("Aggregation complete")
        return 0

    if existing_run:
        # ── Existing run: validate, rerun, or metrics-only ──
        try:
            runner = BenchmarkRunner.from_existing_run(resolved_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1

        # Apply env-dependent values (secrets, urls) from live env onto saved config.
        # Skip strict LLM check when force-rerunning metrics only — the LLM is not needed.
        runner.config.apply_env_overrides(config, strict_llm=not config.force_rerun_metrics)

        # Apply CLI overrides
        runner.config.max_rerun_attempts = config.max_rerun_attempts
        runner.config.force_rerun_metrics = config.force_rerun_metrics
        if config.metrics is not None:
            runner.config.metrics = config.metrics

        dataset_path = runner.config.dataset_path

        try:
            records = EvaluationRecord.load_dataset(dataset_path)
            logger.info(f"Loaded {len(records)} evaluation records")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return 1

        if config.rerun_failed_metrics:
            # ── Rerun only previously failed metric computations ──
            try:
                await runner.rerun_failed_metrics(resolved_dir, records, cli_metrics=config.metrics)
                return 0
            except KeyboardInterrupt:
                logger.warning("Rerun interrupted by user")
                return 130
            except Exception as e:
                logger.error(f"Rerun failed metrics failed: {e}", exc_info=True)
                return 1

        # ── Validate existing + rerun failures ──
        try:
            summary = await runner.validate_existing(
                resolved_dir, records, force_revalidation=config.force_revalidation
            )
            return 0 if summary.failed_records == 0 else 1
        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return 1

    if config.rerun_failed_metrics:
        logger.error("--rerun-failed-metrics requires --run-id pointing to an existing run")
        return 1

    # Load dataset
    try:
        records = EvaluationRecord.load_dataset(config.dataset_path)
        logger.info(f"Loaded {len(records)} evaluation records")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    if not records:
        logger.error("Dataset is empty")
        return 1

    # Dry run - just validate
    if config.dry_run:
        logger.info("Dry run - configuration validated successfully")
        logger.info(f"  Dataset: {len(records)} records")
        if isinstance(config.model, PipelineConfig):
            logger.info(f"  STT model: {config.model.stt}")
            logger.info(f"  LLM model: {config.model.llm}")
            logger.info(f"  TTS model: {config.model.tts}")
        else:
            logger.info(f"  S2S model: {config.model.s2s}")
        logger.info(f"  Max concurrent: {config.max_concurrent_conversations}")
        logger.info(f"  Timeout: {config.conversation_timeout_seconds}s")
        return 0

    # Create and run benchmark
    try:
        runner = BenchmarkRunner(config)
        summary = await runner.run(records)

        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Run ID: {summary.run_id}")
        logger.info(f"  Total records: {summary.total_records}")
        logger.info(f"  Successful: {summary.successful_records}")
        logger.info(f"  Failed: {summary.failed_records}")
        logger.info(f"  Success rate: {summary.success_rate:.1%}")
        logger.info(f"  Duration: {summary.duration_seconds:.1f}s")
        logger.info(f"  Output: {config.output_dir}/{summary.run_id}")

        return 0 if summary.failed_records == 0 else 1

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.warning("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


def _suppress_threading_cleanup_error(unraisable):
    """Suppress Python threading cleanup errors during interpreter shutdown."""
    # Ignore TypeError from threading._DeleteDummyThreadOnDel.__del__
    # which occurs due to race conditions in Python's garbage collector
    if unraisable.exc_type is TypeError and "NoneType" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)
