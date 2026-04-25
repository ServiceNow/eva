"""Validation metrics runner for benchmark validation mode."""

from dataclasses import dataclass, field
from pathlib import Path

from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.models.results import RecordMetrics
from eva.utils.logging import get_logger

logger = get_logger(__name__)

GATE_METRIC = "conversation_valid_end"
LLM_METRICS = ["user_behavioral_fidelity", "user_speech_fidelity"]


@dataclass
class ValidationResult:
    """Result of validating a single record.

    Empty ``failed_metrics`` with ``passed=False`` means the gate rejected the
    record before metrics ran (``not_finished``); a populated list means one or
    more metrics fell below threshold.
    """

    passed: bool
    failed_metrics: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, dict] = field(default_factory=dict)


class ValidationRunner:
    """Two-phase validation: gate-metric filter, then LLM metrics on gate-passed records."""

    VALIDATION_METRICS = [GATE_METRIC] + LLM_METRICS

    def __init__(
        self,
        run_dir: Path,
        dataset: list[EvaluationRecord],
        thresholds: dict[str, float],
        metric_configs: dict[str, dict] | None = None,
        output_ids: list[str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.dataset = dataset
        self.thresholds = thresholds
        self.metric_configs = metric_configs or {}
        self.output_ids = output_ids

    async def run_validation(self) -> dict[str, ValidationResult]:
        validation_results: dict[str, ValidationResult] = {}
        check_ids = self.output_ids if self.output_ids is not None else [r.id for r in self.dataset]
        logger.info(f"Validation: processing {len(check_ids)} records, metrics={self.VALIDATION_METRICS}")
        logger.info(f"Thresholds: {self.thresholds}")

        gate_runner = MetricsRunner(
            run_dir=self.run_dir,
            dataset=self.dataset,
            metric_names=[GATE_METRIC],
            metric_configs=self.metric_configs,
            record_ids=check_ids,
        )
        contexts = gate_runner.process_records()
        gate_run = await gate_runner.run(contexts=contexts)

        gate_passed, not_finished, agent_timeout_ids = self._partition(check_ids, gate_run.all_metrics)
        logger.info(
            f"Gate: {len(gate_passed)} passed ({len(agent_timeout_ids)} agent_timeout_on_user_turn), "
            f"{len(not_finished)} not_finished"
        )

        for record_id in not_finished:
            validation_results[record_id] = ValidationResult(passed=False)

        if gate_passed:
            metrics_runner = MetricsRunner(
                run_dir=self.run_dir,
                dataset=self.dataset,
                metric_names=LLM_METRICS,
                metric_configs=self.metric_configs,
                record_ids=gate_passed,
            )
            passed_contexts = {rid: contexts[rid] for rid in gate_passed if rid in contexts}
            metrics_run = await metrics_runner.run(contexts=passed_contexts)

            for record_id, record_metrics in metrics_run.all_metrics.items():
                vr = self._evaluate_record(record_id, record_metrics, LLM_METRICS)
                vr.scores[GATE_METRIC] = 1.0
                validation_results[record_id] = vr

        passed_count = sum(1 for vr in validation_results.values() if vr.passed)
        total_count = len(validation_results)
        pct = passed_count / total_count * 100 if total_count > 0 else 0.0
        logger.info(f"Validation complete: {passed_count}/{total_count} records passed ({pct:.1f}%)")

        return validation_results

    @staticmethod
    def _partition(
        check_ids: list[str],
        gate_metrics: dict[str, RecordMetrics],
    ) -> tuple[list[str], list[str], set[str]]:
        gate_passed: list[str] = []
        not_finished: list[str] = []
        agent_timeout: set[str] = set()

        for record_id in check_ids:
            rm = gate_metrics.get(record_id)
            ms = rm.metrics.get(GATE_METRIC) if rm else None
            if ms is None or ms.error:
                not_finished.append(record_id)
                continue
            score = ms.normalized_score if ms.normalized_score is not None else ms.score
            if score != 1.0:
                not_finished.append(record_id)
                continue
            gate_passed.append(record_id)
            if ms.details.get("reason") == "agent_timeout_on_user_turn":
                agent_timeout.add(record_id)
        return gate_passed, not_finished, agent_timeout

    def _evaluate_record(
        self,
        record_id: str,
        record_metrics: RecordMetrics,
        metrics_to_check: list[str],
    ) -> ValidationResult:
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

            if metric_score.skipped:
                logger.debug(f"Record {record_id}: Metric '{metric_name}' was skipped")
                continue

            score = metric_score.normalized_score if metric_score.normalized_score is not None else metric_score.score
            scores[metric_name] = score

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
                scores=scores,
                details=details,
            )

        return ValidationResult(passed=True, scores=scores)
