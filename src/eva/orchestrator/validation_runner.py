"""Validation metrics runner for benchmark validation mode."""

from dataclasses import dataclass, field
from pathlib import Path

from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.models.results import RecordMetrics
from eva.utils.conversation_checks import check_conversation_finished
from eva.utils.logging import get_logger
from eva.utils.pass_at_k import parse_trial_record_id

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single record."""

    passed: bool
    failed_metrics: list[str] = field(default_factory=list)
    failure_category: str = "passed"  # "not_finished" | "validation_failed" | "passed"
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, dict] = field(default_factory=dict)  # {metric_name: metric_score.details}


class ValidationRunner:
    """Runs validation metrics to identify failed records."""

    VALIDATION_METRICS = [
        "conversation_finished",
        "user_behavioral_fidelity",
        "user_speech_fidelity",
    ]

    def __init__(
        self,
        run_dir: Path,
        dataset: list[EvaluationRecord],
        thresholds: dict[str, float],
        metric_configs: dict[str, dict] | None = None,
        skip_conversation_finished: bool = False,
        output_ids: list[str] | None = None,
    ):
        """Initialize the validation runner.

        Args:
            run_dir: Directory containing benchmark outputs
            dataset: List of evaluation records (for ground truth)
            thresholds: Validation metric thresholds for pass/fail decisions
            metric_configs: Configuration for specific metrics
            skip_conversation_finished: If True, skip conversation_finished metric
                (used when inline gate already guarantees it passed)
            output_ids: If provided, use these as record directory names instead
                of deriving from dataset record IDs. Needed for nested trial dirs
                (e.g., "1.2.1/trial_0").
        """
        self.run_dir = Path(run_dir)
        self.dataset = dataset
        self.thresholds = thresholds
        self.metric_configs = metric_configs or {}
        self.skip_conversation_finished = skip_conversation_finished
        self.output_ids = output_ids

    async def run_validation(self) -> dict[str, ValidationResult]:
        """Run validation metrics and return results per record.

        When skip_conversation_finished is False (existing output), runs
        conversation_finished first on all records as a fast gate, then only
        runs remaining metrics on records that pass.

        Returns:
            Dict mapping record_id -> ValidationResult
        """
        validation_results: dict[str, ValidationResult] = {}

        if self.skip_conversation_finished:
            # Inline gate already handled conversation_finished.
            # Only run the remaining validation metrics.
            metrics_to_run = [m for m in self.VALIDATION_METRICS if m != "conversation_finished"]
            logger.info(f"Running validation metrics (conversation_finished skipped): {', '.join(metrics_to_run)}")
            logger.info(f"Thresholds: {self.thresholds}")

            record_ids = self.output_ids if self.output_ids is not None else [r.id for r in self.dataset]
            metrics_runner = MetricsRunner(
                run_dir=self.run_dir,
                dataset=self.dataset,
                metric_names=metrics_to_run,
                metric_configs=self.metric_configs,
                record_ids=record_ids,
            )
            metrics_run = await metrics_runner.run()

            for record_id, record_metrics in metrics_run.all_metrics.items():
                vr = self._evaluate_record(record_id, record_metrics, metrics_to_run)
                validation_results[record_id] = vr
                if not vr.passed:
                    logger.info(f"Record {record_id} failed validation: {', '.join(vr.failed_metrics)}")
        else:
            # Full validation: check conversation_finished first as a fast gate
            logger.info(f"Running validation metrics: {', '.join(self.VALIDATION_METRICS)}")
            logger.info(f"Thresholds: {self.thresholds}")

            # Stage 1: Fast conversation_finished check on all records
            records_dir = self.run_dir / "records"
            finished_record_ids: list[str] = []
            not_finished_records: list[str] = []

            check_ids = self.output_ids if self.output_ids is not None else [r.id for r in self.dataset]
            for record_id in check_ids:
                record_dir = records_dir / record_id
                if check_conversation_finished(record_dir):
                    finished_record_ids.append(record_id)
                else:
                    not_finished_records.append(record_id)
                    validation_results[record_id] = ValidationResult(
                        passed=False,
                        failed_metrics=["conversation_finished"],
                        failure_category="not_finished",
                        scores={"conversation_finished": 0.0},
                    )

            if not_finished_records:
                logger.info(
                    f"Short-circuit: {len(not_finished_records)} records failed "
                    f"conversation_finished check, skipping other metrics for them"
                )

            # Stage 2: Run remaining metrics only on records that passed stage 1
            if finished_record_ids:
                remaining_metrics = [m for m in self.VALIDATION_METRICS if m != "conversation_finished"]
                finished_base_ids = {parse_trial_record_id(rid)[0] for rid in finished_record_ids}
                finished_dataset = [r for r in self.dataset if r.id in finished_base_ids]

                metrics_runner = MetricsRunner(
                    run_dir=self.run_dir,
                    dataset=finished_dataset,
                    metric_names=remaining_metrics,
                    metric_configs=self.metric_configs,
                    record_ids=finished_record_ids,
                )
                metrics_run = await metrics_runner.run()

                for record_id, record_metrics in metrics_run.all_metrics.items():
                    vr = self._evaluate_record(record_id, record_metrics, remaining_metrics)
                    # Add conversation_finished score since we know it passed
                    vr.scores["conversation_finished"] = 1.0
                    validation_results[record_id] = vr

        passed_count = sum(1 for vr in validation_results.values() if vr.passed)
        total_count = len(validation_results)
        pct = passed_count / total_count * 100 if total_count > 0 else 0.0
        logger.info(f"Validation complete: {passed_count}/{total_count} records passed ({pct:.1f}%)")

        return validation_results

    def _evaluate_record(
        self,
        record_id: str,
        record_metrics: RecordMetrics,
        metrics_to_check: list[str],
    ) -> ValidationResult:
        """Evaluate a single record against thresholds.

        Args:
            record_id: ID of the record being validated
            record_metrics: Metrics computed for this record
            metrics_to_check: Which metrics to check

        Returns:
            ValidationResult with pass/fail details
        """
        failed_metrics: list[str] = []
        scores: dict[str, float] = {}
        details: dict[str, dict] = {}

        for metric_name in metrics_to_check:
            if metric_name not in record_metrics.metrics:
                logger.warning(
                    f"Record {record_id}: Validation metric '{metric_name}' did not run - considering failed"
                )
                failed_metrics.append(metric_name)
                continue

            metric_score = record_metrics.metrics[metric_name]

            if metric_score.error:
                logger.warning(f"Record {record_id}: Validation metric '{metric_name}' had error: {metric_score.error}")
                failed_metrics.append(metric_name)
                continue

            # Skipped metric (no applicable data) — exclude from validation, not a failure.
            if metric_score.skipped:
                logger.debug(f"Record {record_id}: Metric '{metric_name}' was skipped")
                continue

            score = metric_score.normalized_score if metric_score.normalized_score is not None else metric_score.score
            scores[metric_name] = score

            # Special case: user_speech_fidelity uses per-turn ratings instead of threshold.
            # If any per-turn rating is 1 → fail. Otherwise → pass (skip threshold check).
            if metric_name == "user_speech_fidelity" and metric_score.details:
                per_turn_ratings = metric_score.details.get("per_turn_ratings", {})
                has_low_fidelity = any(r == 1 for r in per_turn_ratings.values() if r is not None)
                if has_low_fidelity:
                    logger.debug(f"Record {record_id}: user_speech_fidelity has per-turn rating of 1")
                    failed_metrics.append(metric_name)
                    if metric_score.details:
                        details[metric_name] = metric_score.details
                continue

            threshold = self.thresholds.get(metric_name, 1.0)
            if score < threshold:
                logger.debug(
                    f"Record {record_id}: Metric '{metric_name}' score {score:.2f} < threshold {threshold:.2f}"
                )
                failed_metrics.append(metric_name)
                if metric_score.details:
                    details[metric_name] = metric_score.details

        if failed_metrics:
            return ValidationResult(
                passed=False,
                failed_metrics=failed_metrics,
                failure_category="validation_failed",
                scores=scores,
                details=details,
            )

        return ValidationResult(
            passed=True,
            failed_metrics=[],
            failure_category="passed",
            scores=scores,
        )
