"""Validation metrics runner for benchmark validation mode."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.models.results import RecordMetrics
from eva.utils.logging import get_logger

logger = get_logger(__name__)

GATE_METRIC = "conversation_valid_end"
LLM_METRICS = ["user_behavioral_fidelity", "user_speech_fidelity"]
# Needs an audio-capable judge (e.g. Gemini); no Bedrock model accepts audio input, so on an
# all-AWS (Bedrock) setup this judge has no runnable model and is skipped (see _audio_judge_available).
AUDIO_JUDGE_METRIC = "user_speech_fidelity"


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
        thresholds: dict[str, float | int],
        metric_configs: dict[str, dict] | None = None,
        output_ids: list[str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.dataset = dataset
        self.thresholds = thresholds
        self.metric_configs = metric_configs or {}
        self.output_ids = output_ids

        # Drop the audio judge when its model isn't a router deployment (it needs an audio-capable
        # model such as Gemini; Bedrock has none). Skipping beats erroring on a model that can't run.
        self._llm_metrics = list(LLM_METRICS)
        if AUDIO_JUDGE_METRIC in self._llm_metrics and not self._audio_judge_available():
            self._llm_metrics = [m for m in LLM_METRICS if m != AUDIO_JUDGE_METRIC]
            logger.warning(
                f"Validation: skipping audio judge '{AUDIO_JUDGE_METRIC}' — its judge model is not a "
                "deployment in EVA_MODEL_LIST (audio judging needs an audio-capable model, e.g. Gemini; "
                "Bedrock has none)."
            )

        # Shared MetricsRunners for validate_one() — lazily initialized on first call.
        # Safe for concurrent calls on different output_ids (asyncio single-threaded).
        self._shared_gate_runner: MetricsRunner | None = None
        self._shared_llm_runner: MetricsRunner | None = None
        self._runner_init_lock = asyncio.Lock()

    def _audio_judge_available(self) -> bool:
        """True if the audio judge's model is a deployment in the LiteLLM router.

        The audio judge resolves its model from ``audio_judge_model`` config or the metric's
        default; if that model isn't in ``EVA_MODEL_LIST`` it can never run (no healthy deployment),
        so validation skips it instead of erroring.
        """
        from eva.metrics.validation.user_speech_fidelity import UserSpeechFidelityMetric
        from eva.utils import router

        cfg = self.metric_configs.get(AUDIO_JUDGE_METRIC, {})
        model = cfg.get("audio_judge_model") or UserSpeechFidelityMetric.default_model
        try:
            deployments = getattr(router.get(), "model_list", []) or []
        except Exception:  # noqa: BLE001 — router not initialized (e.g. some unit tests)
            return False
        return model in {d.get("model_name") for d in deployments}

    async def run_validation(self) -> dict[str, ValidationResult]:
        validation_results: dict[str, ValidationResult] = {}
        check_ids = self.output_ids if self.output_ids is not None else [r.id for r in self.dataset]
        logger.info(f"Validation: processing {len(check_ids)} records, metrics={[GATE_METRIC, *self._llm_metrics]}")
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

        # Separate time-limit-exceeded records from truly not_finished — they get LLM
        # metrics evaluated with gate bypass (same as validate_one(skip_gate=True)).
        time_limit_ids = []
        truly_not_finished = []
        for record_id in not_finished:
            result_path = self.run_dir / "records" / record_id / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                if result_data.get("conversation_ended_reason") == "time_limit_exceeded":
                    time_limit_ids.append(record_id)
                    continue
            truly_not_finished.append(record_id)

        logger.info(
            f"Gate: {len(gate_passed)} passed ({len(agent_timeout_ids)} agent_timeout_on_user_turn), "
            f"{len(truly_not_finished)} not_finished, {len(time_limit_ids)} time_limit_exceeded"
        )

        for record_id in truly_not_finished:
            validation_results[record_id] = ValidationResult(passed=False)

        # Run LLM metrics on gate-passed and time-limit records together.
        # gate_passed means that the record passed all gate metrics.
        # time_limit_ids are those that exceeded the time limit but the user simulation scores passed the gate metrics.
        ids_to_judge = gate_passed + time_limit_ids
        if ids_to_judge:
            metrics_runner = MetricsRunner(
                run_dir=self.run_dir,
                dataset=self.dataset,
                metric_names=self._llm_metrics,
                metric_configs=self.metric_configs,
                record_ids=ids_to_judge,
            )
            all_llm_contexts = {rid: contexts[rid] for rid in ids_to_judge if rid in contexts}
            metrics_run = await metrics_runner.run(contexts=all_llm_contexts)

            gate_passed_set = set(gate_passed)
            for record_id, record_metrics in metrics_run.all_metrics.items():
                vr = self._evaluate_record(record_id, record_metrics, self._llm_metrics)
                if record_id in gate_passed_set:
                    vr.scores[GATE_METRIC] = 1.0
                validation_results[record_id] = vr

        passed_count = sum(1 for vr in validation_results.values() if vr.passed)
        total_count = len(validation_results)
        pct = passed_count / total_count * 100 if total_count > 0 else 0.0
        logger.info(f"Validation complete: {passed_count}/{total_count} records passed ({pct:.1f}%)")

        return validation_results

    async def validate_one(self, output_id: str, *, skip_gate: bool = False) -> ValidationResult:
        """Validate a single record inline for per-record pipelining.

        Runs a two-phase check matching run_validation():
        1. Gate metric (conversation_valid_end) — fast fail if the conversation didn't
           end properly. Returns ValidationResult(passed=False, failed_metrics=[]) which
           signals "not_finished" to the caller (empty failed_metrics convention).
        2. LLM metrics (user_behavioral_fidelity, user_speech_fidelity) — only run if
           the gate passed.

        Both MetricsRunners are lazily initialized on first call and shared across
        concurrent calls — safe because asyncio is single-threaded.

        Args:
            output_id: Record directory name (e.g. "1.2.1" or "1.2.1/trial_0").
            skip_gate: If True, bypass the conversation_valid_end gate and run only
                LLM metrics. Used for time-limit-accepted records where the conversation
                timed out (no goodbye event) but we still want to evaluate against
                thresholds.

        Returns:
            ValidationResult with pass/fail details.
        """
        # Double-checked lazy init for both shared runners.
        if self._shared_gate_runner is None:
            async with self._runner_init_lock:
                if self._shared_gate_runner is None:
                    self._shared_gate_runner = MetricsRunner(
                        run_dir=self.run_dir,
                        dataset=self.dataset,
                        metric_names=[GATE_METRIC],
                        metric_configs=self.metric_configs,
                    )
                    self._shared_llm_runner = MetricsRunner(
                        run_dir=self.run_dir,
                        dataset=self.dataset,
                        metric_names=self._llm_metrics,
                        metric_configs=self.metric_configs,
                    )

        record_dir = self.run_dir / "records" / output_id

        if not skip_gate:
            # Phase 1: gate metric
            gate_metrics = await self._shared_gate_runner.run_and_save_record(output_id, record_dir)
            rm = gate_metrics
            ms = rm.metrics.get(GATE_METRIC) if rm else None
            if ms is None or ms.error:
                return ValidationResult(passed=False)  # empty failed_metrics = "not_finished"
            score = ms.normalized_score if ms.normalized_score is not None else ms.score
            if score != 1.0:
                return ValidationResult(passed=False)  # empty failed_metrics = "not_finished"

        # Phase 2: LLM metrics (gate passed or skipped)
        llm_metrics = await self._shared_llm_runner.run_and_save_record(output_id, record_dir)
        if llm_metrics is None:
            return ValidationResult(passed=False, failed_metrics=list(self._llm_metrics))

        vr = self._evaluate_record(output_id, llm_metrics, self._llm_metrics)
        if not skip_gate:
            vr.scores[GATE_METRIC] = 1.0
        return vr

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

            threshold = float(self.thresholds.get(metric_name, 1.0))
            if score is None or score < threshold:
                score_str = f"{score:.2f}" if score is not None else "None"
                logger.debug(
                    f"Record {record_id}: Metric '{metric_name}' score {score_str} < threshold {threshold:.2f}"
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
