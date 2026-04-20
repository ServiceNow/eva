"""Metrics runner - executes metrics on benchmark outputs."""

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eva.metrics.aggregation import compute_record_aggregates, compute_run_level_aggregates
from eva.metrics.base import BaseMetric, MetricContext
from eva.metrics.processor import MetricsContextProcessor
from eva.metrics.registry import MetricRegistry, get_global_registry
from eva.models.config import PipelineType, get_pipeline_type
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, MetricScore, PassAtKResult, RecordMetrics
from eva.utils.hash_utils import get_dict_hash
from eva.utils.logging import get_logger
from eva.utils.pass_at_k import (
    ATTEMPT_SUFFIX_PATTERN,
    compute_pass_at_k,
    compute_pass_at_k_for_scores,
    parse_trial_record_id,
)
from eva.utils.provenance import capture_metrics_provenance

logger = get_logger(__name__)


@dataclass
class MetricsRunResult:
    """Result of a metrics run, including error information."""

    all_metrics: dict[str, "RecordMetrics"] = field(default_factory=dict)
    total_records: int = 0
    # Per-metric: list of record IDs that failed
    metric_failures: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_metric_failures(self) -> int:
        return sum(len(ids) for ids in self.metric_failures.values())

    @property
    def has_metric_failures(self) -> bool:
        return self.total_metric_failures > 0


class MetricsRunner:
    """Runs metrics on benchmark outputs.

    This is independent of the benchmark runner - it can be run on any
    existing output directory to compute or recompute metrics.

    When multi-trial records are detected (directories named {record_id}_trial_{N}),
    the runner computes standard metrics per attempt and then aggregates pass@k/pass^k
    across attempts for each metric.
    """

    def __init__(
        self,
        run_dir: Path,
        dataset: list[EvaluationRecord],
        metric_names: list[str] | None = None,
        metric_configs: dict[str, dict[str, Any]] | None = None,
        registry: MetricRegistry | None = None,
        num_draws: int = 1,
        record_ids: list[str] | None = None,
        record_metric_filter: dict[str, set[str]] | None = None,
        force_rerun: bool = False,
    ):
        """Initialize the metrics runner.

        Args:
            run_dir: Directory containing benchmark outputs
            dataset: List of evaluation records (for ground truth)
            metric_names: List of metric names to run (None = all registered)
            metric_configs: Configuration for specific metrics
            registry: Metric registry (None = use global)
            num_draws: Number of draws (k) for pass@k computation (default: 1)
            record_ids: If provided, only run metrics on these record directories.
                When None, all record directories are processed (excluding archived attempts).
            record_metric_filter: If provided, maps record_id -> set of metric names to
                recompute. Records not in this map will have their existing metrics read
                from disk without recomputation. Records in this map will only recompute
                the specified metrics and merge with existing results.
            force_rerun: If True, recompute all requested metrics even if they already
                exist and succeeded on disk.
        """
        self.run_dir = Path(run_dir)
        self.dataset = {r.id: r for r in dataset}
        self.record_ids = set(record_ids) if record_ids is not None else None
        self.record_metric_filter = record_metric_filter
        self._is_rerun_mode = record_metric_filter is not None
        self.force_rerun = force_rerun
        self.metric_configs = metric_configs or {}
        self.registry = registry or get_global_registry()

        # pass@k configuration
        self.num_draws = num_draws

        self.metrics_processor = MetricsContextProcessor()

        # Load agent configuration (tools and instructions)
        self._agent_config = self._load_agent_config()

        # Initialize metrics
        if metric_names is None:
            metric_names = self.registry.list_metrics()

        self.metrics: list[BaseMetric] = []
        for name in metric_names:
            config = self.metric_configs.get(name, {})
            metric = self.registry.create(name, config)
            if metric:
                self.metrics.append(metric)
            else:
                logger.warning(f"Metric '{name}' not found, skipping")

        logger.info(f"Metrics runner initialized with {len(self.metrics)} metrics")

    def _load_agent_config(self) -> dict[str, Any]:
        """Load agent configuration from YAML file specified in run config."""
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Run config not found: {config_path}")

        config_data = json.loads(config_path.read_text())

        # Determine pipeline type from config (fallback to False for legacy runs)
        model_data = config_data.get("model", {})
        self._pipeline_type = get_pipeline_type(model_data) if model_data else PipelineType.CASCADE

        agent_config_path = config_data.get("agent_config_path")

        if not agent_config_path:
            raise ValueError("agent_config_path not found in run config")

        # Convert relative path to absolute from project root
        agent_config_path = Path(agent_config_path)
        if not agent_config_path.is_absolute():
            # Assume path is relative to current working directory
            agent_config_path = Path.cwd() / agent_config_path

        if not agent_config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {agent_config_path}")

        with open(agent_config_path) as f:
            agent_config = yaml.safe_load(f)

        # Validate required fields
        if "instructions" not in agent_config:
            raise ValueError(f"Agent config missing 'instructions' field: {agent_config_path}")

        if "tools" not in agent_config:
            raise ValueError(f"Agent config missing 'tools' field: {agent_config_path}")

        if "role" not in agent_config:
            raise ValueError(f"Agent config missing 'role' field: {agent_config_path}")

        return {
            "role": agent_config["role"],
            "instructions": agent_config["instructions"],
            "tools": agent_config["tools"],
        }

    @staticmethod
    def _discover_record_dirs(run_dir: Path, record_ids: set[str] | None = None) -> list[tuple[str, Path]]:
        """Discover record directories under a run, skipping archived attempts.

        Args:
            run_dir: The run output directory.
            record_ids: If provided, only include these record IDs.

        Returns:
            List of (record_id, record_dir_path) tuples.
        """
        records_dir = run_dir / "records"
        if not records_dir.exists():
            raise FileNotFoundError(f"Records directory not found: {records_dir}")

        if record_ids is not None:
            return [(rid, records_dir / rid) for rid in record_ids if (records_dir / rid).is_dir()]

        record_dirs: list[tuple[str, Path]] = []
        for d in records_dir.iterdir():
            if not d.is_dir() or ATTEMPT_SUFFIX_PATTERN.search(d.name):
                continue
            # Check for trial subdirectories (k>1 nested format)
            trial_subdirs = [
                sub
                for sub in sorted(d.iterdir())
                if sub.is_dir() and sub.name.startswith("trial_") and not ATTEMPT_SUFFIX_PATTERN.search(sub.name)
            ]
            if trial_subdirs:
                for sub in trial_subdirs:
                    record_dirs.append((f"{d.name}/{sub.name}", sub))
            else:
                record_dirs.append((d.name, d))  # k=1 record
        return record_dirs

    async def run(self) -> MetricsRunResult:
        """Run all metrics on all records.

        All records and metrics run concurrently. The LiteLLM Router
        manages concurrency per deployment via max_parallel_requests
        and rpm/tpm limits configured in EVA_MODEL_LIST.

        After computing per-record metrics, if multi-attempt records are detected,
        computes pass@k and pass^k aggregation across attempts.

        Returns:
            MetricsRunResult with all metrics and error information
        """
        all_metrics: dict[str, RecordMetrics] = {}

        # Discover ALL record dirs; split into targeted (to compute) and rest (read-only).
        all_record_dirs = self._discover_record_dirs(self.run_dir)
        if self.record_ids is not None:
            targeted = [(rid, rdir) for rid, rdir in all_record_dirs if rid in self.record_ids]
        else:
            targeted = all_record_dirs
        targeted_ids = {rid for rid, _ in targeted}

        # Run targeted records concurrently; LiteLLM limits concurrent API calls.
        tasks = [self._run_and_save_record(rid, rdir) for rid, rdir in targeted]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (record_id, _), result in zip(targeted, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to compute metrics for {record_id}: {result}")
            elif result is not None:
                all_metrics[record_id] = result

        # Include remaining records from disk for globally accurate aggregation.
        for record_id, record_dir in all_record_dirs:
            if record_id in targeted_ids:
                continue
            metrics_path = record_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    all_metrics[record_id] = RecordMetrics.model_validate_json(metrics_path.read_text())
                except Exception as e:
                    logger.warning(f"Failed to read metrics for aggregation ({record_id}): {e}")

        # Compute pass@k if multi-attempt records exist
        pass_at_k_results = self._compute_pass_at_k_from_all_metrics(all_metrics, self.registry, self.num_draws)

        # Save summary (with pass@k data if available)
        metric_failures = await self._save_summary(all_metrics, pass_at_k_results)

        return MetricsRunResult(
            all_metrics=all_metrics,
            total_records=len(targeted),
            metric_failures=metric_failures,
        )

    async def _run_and_save_record(self, record_id: str, record_dir: Path) -> RecordMetrics | None:
        """Run metrics for a record and save results, merging with existing metrics.

        Skips computation when possible:
        - Normal mode: only computes metrics not yet present on disk.
        - Rerun mode: only recomputes metrics that failed; never reruns already-succeeded metrics.
        """
        metrics_path = record_dir / "metrics.json"

        # Read existing metrics from disk if available
        existing_metrics: dict[str, MetricScore] = {}
        if metrics_path.exists():
            try:
                existing_data = json.loads(metrics_path.read_text())
                existing_metrics = {k: MetricScore(**v) for k, v in existing_data.get("metrics", {}).items()}
            except Exception as e:
                logger.warning(f"Failed to read existing metrics for {record_id}: {e}")

        # Rerun mode: record not in filter → return existing as-is
        if self._is_rerun_mode and record_id not in (self.record_metric_filter or {}):
            if existing_metrics:
                record_metrics = RecordMetrics(record_id=record_id, metrics=existing_metrics)
                record_metrics.aggregate_metrics = compute_record_aggregates(record_metrics)
                return record_metrics
            return None

        # Determine which metrics actually need computation
        requested_names = {m.name for m in self.metrics}
        if self._is_rerun_mode:
            # Rerun mode: only recompute filter metrics that haven't already succeeded
            metrics_to_compute = {
                name
                for name in self.record_metric_filter[record_id]
                if name in requested_names and (name not in existing_metrics or existing_metrics[name].error)
            }
        else:
            # Normal mode: compute all requested if force_rerun, otherwise only missing
            if self.force_rerun:
                metrics_to_compute = requested_names
            else:
                metrics_to_compute = requested_names - set(existing_metrics.keys())

        if not metrics_to_compute:
            # All needed metrics already exist on disk
            if existing_metrics:
                record_metrics = RecordMetrics(record_id=record_id, metrics=existing_metrics)
                record_metrics.aggregate_metrics = compute_record_aggregates(record_metrics)
                return record_metrics
            return None

        # Set up filter so _run_record only computes what's needed
        if self.record_metric_filter is None:
            self.record_metric_filter = {}
        self.record_metric_filter[record_id] = metrics_to_compute

        try:
            record_metrics = await self._run_record(record_id, record_dir)

            # Merge with existing metrics
            if existing_metrics:
                merged = {**existing_metrics}
                for name, score in record_metrics.metrics.items():
                    # In normal mode, don't overwrite existing with failed computations
                    if score.error and not self._is_rerun_mode and not self.force_rerun and name in existing_metrics:
                        continue
                    merged[name] = score
                record_metrics = RecordMetrics(
                    record_id=record_metrics.record_id,
                    context=record_metrics.context,
                    metrics=merged,
                )

            # Compute EVA composite aggregates
            record_metrics.aggregate_metrics = compute_record_aggregates(record_metrics)

            metrics_path.write_text(record_metrics.model_dump_json(indent=2))

            return record_metrics
        except Exception as e:
            logger.error(f"Failed to compute metrics for {record_id}: {e}", exc_info=True)
            raise

    async def _run_record(self, record_id: str, record_dir: Path) -> RecordMetrics:
        """Run all metrics for a single record in parallel."""
        logger.debug(f"Computing metrics for record: {record_id}")

        # Load conversation data
        context = self._load_context(record_id, record_dir)

        # Determine which metrics to run for this record
        metrics_to_run = self.metrics
        if self.record_metric_filter and record_id in self.record_metric_filter:
            allowed = self.record_metric_filter[record_id]
            metrics_to_run = [m for m in self.metrics if m.name in allowed]

        # Skip record entirely if no conversation turns
        if context.num_assistant_turns == 0 or context.num_user_turns == 0:
            logger.warning(f"Skipping record {record_id}: no assistant or user turns found")
            return RecordMetrics(
                record_id=record_id,
                context=context.to_dict(),
                metrics={
                    metric.name: MetricScore(
                        name=metric.name,
                        score=0.0,
                        normalized_score=0.0,
                        error="Skipped: no assistant or user turns",
                    )
                    for metric in metrics_to_run
                },
            )

        # Create tasks for all metrics
        async def compute_metric(metric: BaseMetric) -> tuple[str, MetricScore]:
            """Compute a single metric and handle errors."""
            try:
                logger.info(f"[{record_id}] Starting metric: {metric.name}")
                score = await metric.compute(context)
                logger.info(
                    f"[{record_id}] Finished metric: {metric.name} "
                    f"(score={score.score}, normalized={score.normalized_score}, error={score.error})"
                )
                return metric.name, score
            except Exception as e:
                logger.error(f"[{record_id}] Metric {metric.name} failed: {e}")
                return metric.name, MetricScore(
                    name=metric.name,
                    score=0.0,
                    error=str(e),
                )

        # Filter out metrics incompatible with the pipeline type
        applicable_metrics = metrics_to_run
        if context.is_audio_native:
            skipped = [m.name for m in metrics_to_run if m.skip_audio_native]
            if skipped:
                logger.info(f"[{record_id}] Skipping metrics incompatible with audio-native pipeline: {skipped}")
            applicable_metrics = [m for m in metrics_to_run if not m.skip_audio_native]

        # Run all metrics in parallel
        tasks = [compute_metric(metric) for metric in applicable_metrics]
        results = await asyncio.gather(*tasks)

        # Build metric scores dictionary
        metric_scores: dict[str, MetricScore] = dict(results)

        # Serialize context to dict for storage
        context_dict = context.to_dict()

        return RecordMetrics(record_id=record_id, context=context_dict, metrics=metric_scores)

    def _load_context(self, record_id: str, record_dir: Path) -> MetricContext:
        """Load all data needed for metric computation."""
        # Strip _trial_N suffix to get base record ID for dataset lookup.
        base_record_id, _ = parse_trial_record_id(record_id)

        # Get ground truth from dataset using the base record ID
        record = self.dataset.get(base_record_id)
        if record is None:
            raise ValueError(f"Record {record_id} (base: {base_record_id}) not found in dataset")

        gt = record.ground_truth

        # Load conversation result
        result_path = record_dir / "result.json"
        result_data = {}
        if result_path.exists():
            result_data = json.loads(result_path.read_text())

        # Create ConversationResult object
        result = ConversationResult(**result_data)

        # Use postprocessor to process logs and create enriched context
        metrics_context = self.metrics_processor.process_record(result, record_dir, pipeline_type=self._pipeline_type)

        # Get agent instructions and tools from config
        agent_instructions = self._agent_config["instructions"]
        agent_tools = self._agent_config["tools"]
        agent_role = self._agent_config["role"]

        if record.agent_override and record.agent_override.instructions:
            agent_instructions = record.agent_override.instructions

        user_persona = record.user_config["user_persona"]

        # Load scenario database states (REQUIRED for deterministic metrics)
        initial_db_path = record_dir / "initial_scenario_db.json"
        final_db_path = record_dir / "final_scenario_db.json"

        if not initial_db_path.exists():
            raise FileNotFoundError(
                f"Initial scenario database not found at {initial_db_path}. "
                "This is required for deterministic task completion metrics."
            )
        if not final_db_path.exists():
            raise FileNotFoundError(
                f"Final scenario database not found at {final_db_path}. "
                "This is required for deterministic task completion metrics."
            )

        initial_scenario_db = json.loads(initial_db_path.read_text())
        final_scenario_db = json.loads(final_db_path.read_text())

        # Get hashes from result or compute if needed
        initial_scenario_db_hash = getattr(result, "initial_scenario_db_hash", None) or get_dict_hash(
            initial_scenario_db
        )
        final_scenario_db_hash = getattr(result, "final_scenario_db_hash", None) or get_dict_hash(final_scenario_db)

        # Get expected DB from ground truth (REQUIRED)
        if not hasattr(gt, "expected_scenario_db") or not gt.expected_scenario_db:
            raise ValueError(
                f"Record {record_id} missing expected_scenario_db in ground_truth. "
                "This is required for deterministic task completion metrics."
            )

        # Copy all shared attributes from _ProcessorContext to MetricContext
        postprocessor_fields = {}
        if metrics_context:
            metric_context_params = set(inspect.signature(MetricContext).parameters)
            for attr in vars(metrics_context):
                if attr in metric_context_params:
                    postprocessor_fields[attr] = getattr(metrics_context, attr)

        # Create MetricContext: postprocessor fields + fields only available here
        postprocessor_fields.setdefault("record_id", record_id)
        metric_context = MetricContext(
            **postprocessor_fields,
            # Ground truth (only in dataset, not in postprocessor)
            user_goal=record.user_goal,
            user_persona=user_persona,
            # Scenario database state (loaded from files above)
            expected_scenario_db=gt.expected_scenario_db,
            initial_scenario_db=initial_scenario_db,
            final_scenario_db=final_scenario_db,
            initial_scenario_db_hash=initial_scenario_db_hash,
            final_scenario_db_hash=final_scenario_db_hash,
            # Agent config (from YAML)
            agent_role=agent_role,
            agent_instructions=agent_instructions,
            agent_tools=agent_tools,
            current_date_time=record.current_date_time,
            # Basic stats from result
            num_turns=result.num_turns,
            tools_called=result.tools_called,
            duration_seconds=result.duration_seconds,
            # Paths
            output_dir=str(record_dir),
        )
        return metric_context

    @staticmethod
    def _aggregate_scores(
        scores: list[float],
        total_records: int,
        error_count: int = 0,
        missing_count: int = 0,
    ) -> dict[str, Any]:
        """Compute standard aggregate stats from a list of scores."""
        none_count = error_count + missing_count
        return {
            "mean": round(sum(scores) / len(scores), 4) if scores else None,
            "min": round(min(scores), 4) if scores else None,
            "max": round(max(scores), 4) if scores else None,
            "count": len(scores),
            "none_count": none_count,
            "error_count": error_count,
            "missing_count": missing_count,
            "total_records": total_records,
        }

    @staticmethod
    def _build_per_metric_aggregates(
        all_metrics: dict[str, RecordMetrics],
        metric_names: list[str],
        pass_at_k_results: dict[str, dict[str, PassAtKResult]] | None = None,
        num_draws: int = 1,
    ) -> dict[str, dict[str, Any]]:
        """Build per-metric aggregate stats including pass_k.

        Args:
            all_metrics: All computed metrics keyed by record ID.
            metric_names: List of metric names to aggregate.
            pass_at_k_results: Per-record pass@k results (if multi-trial).
            num_draws: Number of draws (k) for pass@k.

        Returns:
            Dict mapping metric name to aggregate stats.
        """
        total_records = len(all_metrics)
        metric_aggregates: dict[str, dict[str, Any]] = {}
        for name in metric_names:
            scores: list[float] = []
            error_count = 0
            missing_count = 0
            # Per-turn None tracking (for metrics that report num_turns / num_evaluated)
            total_turns_across_records = 0
            total_evaluated_across_records = 0
            total_not_applicable_across_records = 0
            records_with_turn_nones = 0

            for record_metrics in all_metrics.values():
                if name not in record_metrics.metrics:
                    missing_count += 1
                    continue
                score = record_metrics.metrics[name]
                if score.error is not None:
                    error_count += 1
                    continue
                value = score.normalized_score if score.normalized_score is not None else score.score
                if value is not None:
                    scores.append(value)
                else:
                    missing_count += 1

                # Aggregate per-turn None stats from details
                details = score.details or {}
                num_turns = details.get("num_turns")
                num_evaluated = details.get("num_evaluated")
                if num_turns is not None and num_evaluated is not None:
                    total_turns_across_records += num_turns
                    total_evaluated_across_records += num_evaluated
                    total_not_applicable_across_records += details.get("num_not_applicable", 0) or 0
                    if num_evaluated < num_turns:
                        records_with_turn_nones += 1

            none_count = error_count + missing_count
            if scores or none_count > 0:
                entry = MetricsRunner._aggregate_scores(scores, total_records, error_count, missing_count)

                if total_turns_across_records > 0:
                    none_turns = total_turns_across_records - total_evaluated_across_records
                    coverage: dict[str, Any] = {
                        "total_turns": total_turns_across_records,
                        "evaluated_turns": total_evaluated_across_records,
                        "none_turns": none_turns,
                        "none_turn_rate": round(none_turns / total_turns_across_records, 4),
                        "records_with_none_turns": records_with_turn_nones,
                    }
                    if total_not_applicable_across_records > 0:
                        coverage["not_applicable_turns"] = total_not_applicable_across_records
                    entry["per_turn_coverage"] = coverage

                metric_aggregates[name] = entry

        # Add pass_k aggregates if available
        if pass_at_k_results:
            for name in metric_aggregates:
                pass_at_1_values: list[float] = []
                pass_at_k_values: list[float] = []
                pass_power_k_obs_values: list[float] = []
                pass_power_k_theo_values: list[float] = []

                for record_pass_at_k in pass_at_k_results.values():
                    if name in record_pass_at_k:
                        result = record_pass_at_k[name]
                        pass_at_1_values.append(compute_pass_at_k(result.n, result.c, 1))
                        pass_at_k_values.append(result.pass_at_k)
                        pass_power_k_obs_values.append(result.pass_power_k)
                        theoretical = (result.c / result.n) ** result.k if result.n > 0 else 0.0
                        pass_power_k_theo_values.append(theoretical)

                if pass_at_k_values:
                    count = len(pass_at_k_values)
                    metric_aggregates[name]["pass_k"] = {
                        "pass_at_1": round(sum(pass_at_1_values) / count, 4),
                        "pass_at_k": round(sum(pass_at_k_values) / count, 4),
                        "pass_power_k_observed": round(sum(pass_power_k_obs_values) / count, 4),
                        "pass_power_k_theoretical": round(sum(pass_power_k_theo_values) / count, 4),
                        "k": num_draws,
                        "count": count,
                    }

        # Generic sub-metric aggregation
        for name in metric_aggregates:
            all_sub_keys: set[str] = set()
            for record_metrics in all_metrics.values():
                ms = record_metrics.metrics.get(name)
                if ms and ms.sub_metrics:
                    all_sub_keys.update(ms.sub_metrics.keys())

            if not all_sub_keys:
                continue

            sub_aggs: dict[str, dict[str, Any]] = {}
            for sub_key in sorted(all_sub_keys):
                sub_scores: list[float] = []
                sub_missing = 0
                for record_metrics in all_metrics.values():
                    ms = record_metrics.metrics.get(name)
                    if ms is None or ms.error is not None:
                        sub_missing += 1
                        continue
                    sub_ms = (ms.sub_metrics or {}).get(sub_key)
                    if sub_ms is not None and sub_ms.score is not None:
                        sub_scores.append(
                            sub_ms.normalized_score if sub_ms.normalized_score is not None else sub_ms.score
                        )
                    else:
                        sub_missing += 1

                if sub_scores or sub_missing > 0:
                    sub_aggs[sub_key] = MetricsRunner._aggregate_scores(sub_scores, total_records, 0, sub_missing)

            if sub_aggs:
                metric_aggregates[name]["sub_metrics"] = sub_aggs

        return metric_aggregates

    @staticmethod
    def _build_data_quality(
        all_metrics: dict[str, RecordMetrics],
        metric_aggregates: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build cross-metric data quality summary tracking None scores and per-turn gaps."""
        records_with_any_none: set[str] = set()
        records_with_errors: set[str] = set()
        records_with_missing: set[str] = set()
        metrics_with_none: dict[str, int] = {}
        metrics_with_errors: dict[str, int] = {}
        metrics_with_missing: dict[str, int] = {}
        metrics_with_per_turn_nones: dict[str, dict[str, Any]] = {}

        for name, agg in metric_aggregates.items():
            agg_error_count = agg.get("error_count", 0)
            agg_missing_count = agg.get("missing_count", 0)

            if agg_error_count > 0:
                metrics_with_errors[name] = agg_error_count
            if agg_missing_count > 0:
                metrics_with_missing[name] = agg_missing_count

            if agg.get("none_count", 0) > 0:
                metrics_with_none[name] = agg["none_count"]
                for record_id, record_metrics in all_metrics.items():
                    if name not in record_metrics.metrics:
                        records_with_any_none.add(record_id)
                        records_with_missing.add(record_id)
                    elif record_metrics.metrics[name].error is not None:
                        records_with_any_none.add(record_id)
                        records_with_errors.add(record_id)
                    else:
                        score = record_metrics.metrics[name]
                        value = score.normalized_score if score.normalized_score is not None else score.score
                        if value is None:
                            records_with_any_none.add(record_id)
                            records_with_missing.add(record_id)

            coverage = agg.get("per_turn_coverage")
            if coverage and coverage["none_turns"] > 0:
                metrics_with_per_turn_nones[name] = {
                    "none_turn_rate": coverage["none_turn_rate"],
                    "none_turns": coverage["none_turns"],
                    "total_turns": coverage["total_turns"],
                    "records_affected": coverage["records_with_none_turns"],
                }

        data_quality: dict[str, Any] = {
            "records_with_any_none": len(records_with_any_none),
            "records_with_errors": len(records_with_errors),
            "records_with_missing": len(records_with_missing),
            "total_records": len(all_metrics),
            "metrics_with_none_scores": metrics_with_none,
            "metrics_with_errors": metrics_with_errors,
            "metrics_with_missing": metrics_with_missing,
        }
        if metrics_with_per_turn_nones:
            data_quality["metrics_with_per_turn_nones"] = metrics_with_per_turn_nones

        return data_quality

    @staticmethod
    def _compute_pass_at_k_from_all_metrics(
        all_metrics: dict[str, RecordMetrics],
        registry: MetricRegistry,
        num_draws: int,
    ) -> dict[str, dict[str, PassAtKResult]]:
        """Compute pass@k for multi-trial records using registry for metric metadata.

        Args:
            all_metrics: All computed metrics keyed by record ID.
            registry: Metric registry for threshold/exclusion info.
            num_draws: Number of draws (k).

        Returns:
            Dict mapping base_record_id -> {metric_name -> PassAtKResult}.
        """
        # Group by base record ID
        grouped: dict[str, list[tuple[int, RecordMetrics]]] = {}
        for record_id, metrics in all_metrics.items():
            base_id, trial_idx = parse_trial_record_id(record_id)
            if trial_idx is not None:
                grouped.setdefault(base_id, []).append((trial_idx, metrics))

        if not grouped:
            return {}

        logger.info(f"Computing pass@k (k={num_draws}) for {len(grouped)} records with multi-trial data")

        # Collect all metric names across records
        all_metric_names: set[str] = set()
        for record_metrics in all_metrics.values():
            all_metric_names.update(record_metrics.metrics.keys())

        results: dict[str, dict[str, PassAtKResult]] = {}

        for base_id, attempts in grouped.items():
            attempts.sort(key=lambda x: x[0])
            record_results: dict[str, PassAtKResult] = {}

            for metric_name in all_metric_names:
                # Use registry to get metadata if available
                metric_cls = registry.get(metric_name)
                if metric_cls is not None:
                    if metric_cls.exclude_from_pass_at_k:
                        continue
                    threshold = metric_cls.pass_at_k_threshold
                else:
                    threshold = 0.5  # default

                attempt_scores: list[MetricScore] = []
                for _, record_metrics in attempts:
                    if metric_name in record_metrics.metrics:
                        attempt_scores.append(record_metrics.metrics[metric_name])

                if not attempt_scores:
                    continue

                result = compute_pass_at_k_for_scores(
                    metric_name, attempt_scores, threshold, min(num_draws, len(attempt_scores))
                )
                if result is not None:
                    record_results[metric_name] = result

            if record_results:
                results[base_id] = record_results

        logger.info(f"pass@k computation complete for {len(results)} records")
        return results

    async def _save_summary(
        self,
        all_metrics: dict[str, RecordMetrics],
        pass_at_k_results: dict[str, dict[str, PassAtKResult]] | None = None,
    ) -> dict[str, list[str]]:
        """Save metrics summary, including pass@k data if available.

        Returns:
            Dict mapping metric name to list of record IDs that failed for that metric.
        """
        if not all_metrics:
            return {}

        run_metric_names = [m.name for m in self.metrics]
        # Aggregate per_metric for ALL metrics present across records (not just those just run),
        # so that a partial re-run (e.g. --metrics response_speed) preserves other metrics.
        all_metric_names = sorted({name for rm in all_metrics.values() for name in rm.metrics})
        metric_aggregates = self._build_per_metric_aggregates(
            all_metrics, all_metric_names, pass_at_k_results, self.num_draws
        )

        # Compute metric failures for MetricsRunResult (only for metrics just run)
        metric_failures: dict[str, list[str]] = {}
        for name in run_metric_names:
            for record_id, record_metrics in all_metrics.items():
                if name in record_metrics.metrics:
                    score = record_metrics.metrics[name]
                    if score.error is not None:
                        metric_failures.setdefault(name, []).append(record_id)

        # Compute EVA composite run-level aggregates
        overall_scores = compute_run_level_aggregates(all_metrics, self.num_draws)

        # Load existing summary to preserve fields for metrics not being re-run
        summary_path = self.run_dir / "metrics_summary.json"
        existing_summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                existing_summary = json.loads(summary_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to read existing metrics_summary.json: {e}")

        # Merge metric_errors: preserve existing errors for metrics not being re-run
        merged_metric_errors: dict[str, dict[str, Any]] = dict(existing_summary.get("metric_errors") or {})
        for metric_name, failed_record_ids in metric_failures.items():
            merged_metric_errors[metric_name] = {
                "failed_count": len(failed_record_ids),
                "total_count": len(all_metrics),
                "failed_records": failed_record_ids,
            }
        # Remove error entries for metrics that are now in run_metric_names but had no failures
        for name in run_metric_names:
            if name not in metric_failures:
                merged_metric_errors.pop(name, None)

        data_quality = self._build_data_quality(all_metrics, metric_aggregates)

        summary: dict[str, Any] = {
            "total_records": len(all_metrics),
            "data_quality": data_quality,
            "overall_scores": overall_scores,
            "per_metric": metric_aggregates,
        }

        if merged_metric_errors:
            summary["metric_errors"] = merged_metric_errors

        # Add pass@k configuration if applicable
        if pass_at_k_results:
            summary["pass_at_k_config"] = {
                "num_draws": self.num_draws,
                "per_metric_thresholds": {
                    m.name: m.pass_at_k_threshold for m in self.metrics if not m.exclude_from_pass_at_k
                },
                "exclude_metrics": sorted(m.name for m in self.metrics if m.exclude_from_pass_at_k),
            }
        elif existing_summary.get("pass_at_k_config"):
            summary["pass_at_k_config"] = existing_summary["pass_at_k_config"]

        try:
            run_config = json.loads((self.run_dir / "config.json").read_text())
            provenance = capture_metrics_provenance(run_metric_names, run_config=run_config)
            summary["provenance"] = provenance.model_dump(mode="json")
        except Exception as e:
            logger.warning(f"Failed to capture metrics provenance: {e}")

        summary_path.write_text(json.dumps(summary, indent=2))

        logger.info(f"Metrics summary saved to {summary_path}")
        logger.info("Metrics summary:")
        logger.info(json.dumps(summary, indent=2))

        return metric_failures

    @classmethod
    async def run_aggregate_only(cls, run_dir: Path, num_draws: int = 1) -> None:
        """Recompute EVA aggregate scores from existing metrics.json files.

        No dataset, agent config, or LLM calls needed. Reads each record's
        metrics.json, computes aggregate_metrics, writes them back, and
        updates metrics_summary.json with overall_scores.

        Args:
            run_dir: Path to the run output directory.
            num_draws: Number of draws (k) for pass@k computation.
        """
        run_dir = Path(run_dir)
        record_dirs = cls._discover_record_dirs(run_dir)

        all_metrics: dict[str, RecordMetrics] = {}

        for record_id, record_path in record_dirs:
            metrics_path = record_path / "metrics.json"
            if not metrics_path.exists():
                logger.warning(f"No metrics.json for {record_id}, skipping")
                continue

            record_metrics = RecordMetrics.model_validate_json(metrics_path.read_text())
            record_metrics.aggregate_metrics = compute_record_aggregates(record_metrics)
            metrics_path.write_text(record_metrics.model_dump_json(indent=2))
            all_metrics[record_id] = record_metrics

        if not all_metrics:
            logger.warning("No records with metrics found")
            return

        registry = get_global_registry()

        # Compute pass@k if multi-trial records exist
        pass_at_k_results = cls._compute_pass_at_k_from_all_metrics(all_metrics, registry, num_draws)

        # Collect all metric names across records
        all_metric_names = sorted({name for rm in all_metrics.values() for name in rm.metrics})

        # Compute per-metric aggregates (including pass_k)
        metric_aggregates = cls._build_per_metric_aggregates(
            all_metrics, all_metric_names, pass_at_k_results or None, num_draws
        )

        # Compute run-level aggregates
        overall_scores = compute_run_level_aggregates(all_metrics, num_draws)

        # Update metrics_summary.json (preserve existing fields, replace computed sections)
        summary_path = run_dir / "metrics_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {"total_records": len(all_metrics)}

        summary.pop("overall_pass_fail", None)
        summary["total_records"] = len(all_metrics)
        summary["data_quality"] = cls._build_data_quality(all_metrics, metric_aggregates)
        summary["overall_scores"] = overall_scores
        summary["per_metric"] = metric_aggregates

        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Aggregate-only complete: {len(all_metrics)} records updated, summary at {summary_path}")
