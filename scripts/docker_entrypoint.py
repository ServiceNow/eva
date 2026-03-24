#!/usr/bin/env python3
"""Docker entrypoint script that runs the benchmark."""

import asyncio
import os
import sys
from pathlib import Path

from eva.models.config import RunConfig
from eva.models.record import EvaluationRecord
from eva.orchestrator.runner import BenchmarkRunner
from eva.utils import router
from eva.utils.logging import get_logger, setup_logging


def _load_dataset(dataset_path: Path) -> list[EvaluationRecord] | None:
    """Load evaluation records from dataset file. Returns None on failure."""
    logger = get_logger(__name__)
    try:
        records = EvaluationRecord.load_dataset(dataset_path)
        logger.info(f"Loaded {len(records)} evaluation records")
        if not records:
            logger.error("Dataset is empty")
            return None
        return records
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


async def main() -> int:
    """Main entrypoint — always runs benchmark via BenchmarkRunner."""
    config = RunConfig(_env_file=".env")

    output_folder = os.environ.get("EVA_OUTPUT_FOLDER")
    if output_folder:
        config.output_dir = config.output_dir / output_folder

    router.init(config.model_list)

    setup_logging(level=config.log_level)
    logger = get_logger(__name__)

    logger.info("Starting EVA benchmark")

    records = _load_dataset(config.dataset_path)
    if records is None:
        return 1

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

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
