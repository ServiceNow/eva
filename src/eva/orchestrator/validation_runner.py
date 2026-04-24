"""Validation metrics runner for benchmark validation mode."""

from dataclasses import dataclass, field
from pathlib import Path

from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.models.results import RecordMetrics
from eva.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single record.

    Semantics are encoded in two fields:

      - ``passed``: True when the record should count as a benchmark success.
      - ``failed_metrics``: empty when the gate rejected the record before metrics ran
        (i.e. "not finished"); populated with the names of metrics below threshold when
        metrics ran and some failed. Callers that need to distinguish the two failure
        modes (e.g. for ``rerun_history`` bookkeeping) check ``failed_metrics`` directly.

    Agent-timeout-on-user-turn records pass the gate with ``passed=True`` — the
    agent-side failure is surfaced via the ``conversation_correctly_finished`` diagnostic metric
    in ``metrics.json``, not through this object.
    """

    passed: bool
    failed_metrics: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, dict] = field(default_factory=dict)


class ValidationRunner:
    """Runs the validation gate and validation metrics.

    The gate is a pure classifier over :class:`_ProcessorContext` objects:
      - ``conversation_valid_end`` → gate passes, metrics run
      - ``agent_timeout_on_user_turn`` (property) → gate passes, metrics run; the
        record is treated as passed at the validation layer (the agent failure is
        surfaced in metrics.json, not as a validation failure)
      - anything else → ``not_finished`` (retry-worthy)

    A single :class:`MetricsRunner` is constructed up front; its ``process_records``
    phase produces the contexts the gate classifies, and those same contexts are fed
    back into ``run`` so the processor executes once per record.
    """

    VALIDATION_METRICS = [
        "conversation_valid_end",
        "user_behavioral_fidelity",
        "user_speech_fidelity",
    ]

    def __init__(
        self,
        run_dir: Path,
        dataset: list[EvaluationRecord],
        thresholds: dict[str, float],
        metric_configs: dict[str, dict] | None = None,
        output_ids: list[str] | None = None,
    ):
        """Initialize the validation runner.

        Args:
            run_dir: Directory containing benchmark outputs
            dataset: List of evaluation records (for ground truth)
            thresholds: Validation metric thresholds for pass/fail decisions
            metric_configs: Configuration for specific metrics
            output_ids: If provided, use these as record directory names instead
                of deriving from dataset record IDs. Needed for nested trial dirs
                (e.g., "1.2.1/trial_0").
        """
        self.run_dir = Path(run_dir)
        self.dataset = dataset
        self.thresholds = thresholds
        self.metric_configs = metric_configs or {}
        self.output_ids = output_ids

    @staticmethod
    def _not_finished_result() -> ValidationResult:
        # Gate rejected: no metrics ran. Empty failed_metrics distinguishes this from
        # the validation_failed case (where failed_metrics carries threshold violators).
        return ValidationResult(passed=False)

    @staticmethod
    def _classify(
        contexts: dict[str, object],
        check_ids: list[str],
    ) -> tuple[list[str], list[str], set[str]]:
        """Sort records into (gate_passed, not_finished, agent_timeout) buckets.

        Records missing from ``contexts`` (processor failure, missing result.json) are
        treated as ``not_finished``. The ``conversation_valid_end`` context attribute is
        already the OR of "ended with goodbye" and "agent timed out on user turn", so
        it alone decides gate-pass. ``agent_timeout`` is tracked separately so callers
        can flag those records as terminal (no rerun) even though they gate-pass.
        """
        gate_passed: list[str] = []
        not_finished: list[str] = []
        agent_timeout: set[str] = set()

        for record_id in check_ids:
            ctx = contexts.get(record_id)
            if ctx is None:
                not_finished.append(record_id)
                continue
            if ctx.conversation_valid_end:  # type: ignore[attr-defined]
                gate_passed.append(record_id)
                if is_agent_timeout_on_user_turn(
                    ctx.conversation_ended_reason,  # type: ignore[attr-defined]
                    ctx.audio_timestamps_user_turns,  # type: ignore[attr-defined]
                    ctx.audio_timestamps_assistant_turns,  # type: ignore[attr-defined]
                ):
                    agent_timeout.add(record_id)
                continue
            not_finished.append(record_id)
        return gate_passed, not_finished, agent_timeout

    async def run_validation(self) -> dict[str, ValidationResult]:
        """Run the gate then validation metrics; return a result per record."""
        validation_results: dict[str, ValidationResult] = {}
        check_ids = self.output_ids if self.output_ids is not None else [r.id for r in self.dataset]

        metrics_to_run = [m for m in self.VALIDATION_METRICS if m != "conversation_valid_end"]
        logger.info(f"Validation: processing {len(check_ids)} records, metrics={metrics_to_run}")
        logger.info(f"Thresholds: {self.thresholds}")

        metrics_runner = MetricsRunner(
            run_dir=self.run_dir,
            dataset=self.dataset,
            metric_names=metrics_to_run,
            metric_configs=self.metric_configs,
            record_ids=check_ids,
        )

        contexts = metrics_runner.process_records()
        gate_passed, not_finished, agent_timeout_ids = self._classify(contexts, check_ids)

        logger.info(
            f"Gate: {len(gate_passed)} passed ({len(agent_timeout_ids)} agent_timeout_on_user_turn), "
            f"{len(not_finished)} not_finished"
        )

        for record_id in not_finished:
            validation_results[record_id] = self._not_finished_result()

        if gate_passed:
            # Narrow the existing runner to gate-passed records and feed it the already-
            # computed contexts so the processor runs exactly once per record.
            metrics_runner.record_ids = set(gate_passed)
            passed_contexts = {rid: contexts[rid] for rid in gate_passed}
            metrics_run = await metrics_runner.run(contexts=passed_contexts)

            for record_id, record_metrics in metrics_run.all_metrics.items():
                vr = self._evaluate_record(record_id, record_metrics, metrics_to_run)
                vr.scores["conversation_valid_end"] = 1.0
                if record_id in agent_timeout_ids:
                    # Agent-side failure; surfaced via the conversation_correctly_finished diagnostic
                    # metric in metrics.json. Validation produced usable data, so the
                    # record passes the gate — the agent bug is not a validation failure.
                    vr.passed = True
                    vr.failed_metrics = []
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
        """Evaluate a single record against thresholds."""
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
