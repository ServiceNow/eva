"""Result and metrics data models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorDetails(BaseModel):
    """Detailed error information."""

    error_type: str = Field(
        ...,
        description="Error type: llm_error, tts_error, stt_error, tool_error, system_error, network_error, timeout_error",
    )
    error_source: str = Field(
        ..., description="Error source: openai, cartesia, deepgram, tool_executor, port_pool, etc."
    )
    is_retryable: bool = Field(..., description="Whether this error can be retried")
    retry_count: int = Field(0, description="Number of retry attempts")
    retry_succeeded: bool = Field(False, description="Whether retry succeeded")
    timestamps: list[str] = Field(default_factory=list, description="Timestamp of each attempt")
    stack_trace: str | None = Field(None, description="Stack trace if available")
    original_error: str = Field(..., description="Original error message")


class LatencyStats(BaseModel):
    """Latency statistics for a component."""

    mean_ms: float = Field(..., description="Mean latency in milliseconds")
    p50_ms: float = Field(..., description="Median latency in milliseconds")
    p95_ms: float = Field(..., description="95th percentile latency in milliseconds")
    p99_ms: float = Field(..., description="99th percentile latency in milliseconds")
    total_calls: int = Field(..., description="Total number of calls")


class ConversationResult(BaseModel):
    """Result of a single conversation."""

    record_id: str = Field(..., description="ID of the evaluation record")
    completed: bool = Field(..., description="Whether the conversation completed successfully")
    error: str | None = Field(None, description="Error message if failed")
    error_details: ErrorDetails | None = Field(
        None, description="Detailed error information (new field, optional for backward compatibility)"
    )

    # Latency statistics
    llm_latency: LatencyStats | None = Field(None, description="LLM latency statistics")
    stt_latency: LatencyStats | None = Field(None, description="STT latency statistics")
    tts_latency: LatencyStats | None = Field(None, description="TTS latency statistics")

    # Timing
    started_at: datetime = Field(..., description="When the conversation started")
    ended_at: datetime = Field(..., description="When the conversation ended")
    duration_seconds: float = Field(..., description="Total duration in seconds")

    # Paths to outputs
    output_dir: str = Field(..., description="Path to output directory for this record")
    audio_assistant_path: str | None = Field(None, description="Path to assistant audio file")
    audio_user_path: str | None = Field(None, description="Path to user audio file")
    audio_mixed_path: str | None = Field(None, description="Path to mixed audio file")
    transcript_path: str | None = Field(None, description="Path to transcript JSONL file")
    audit_log_path: str | None = Field(None, description="Path to audit log JSON file")
    conversation_log_path: str | None = Field(None, description="Path to conversation log file")
    pipecat_logs_path: str | None = Field(None, description="Path to pipecat logs JSONL file")
    elevenlabs_logs_path: str | None = Field(None, description="Path to elevenlabs logs JSONL file")

    # Summary stats (pre-metrics)
    num_turns: int = Field(0, description="Number of conversation turns")
    num_tool_calls: int = Field(0, description="Number of tool calls made")
    tools_called: list[str] = Field(default_factory=list, description="List of tools that were called")
    conversation_ended_reason: str | None = Field(
        None,
        description="Reason conversation ended: 'goodbye', 'timeout', 'transfer', 'error'",
    )
    initial_scenario_db_hash: str | None = Field(None, description="SHA-256 hash of initial scenario database")
    final_scenario_db_hash: str | None = Field(None, description="SHA-256 hash of final scenario database")


class MetricScore(BaseModel):
    """Score for a single metric."""

    name: str = Field(..., description="Metric name")
    score: float = Field(..., description="Raw score value")
    normalized_score: float | None = Field(None, description="Normalized score (0-1 scale)")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional metric details")
    error: str | None = Field(None, description="Error message if metric computation failed")


class PassAtKResult(BaseModel):
    """pass@k and pass^k result for a single metric across multiple trials."""

    metric_name: str = Field(..., description="Name of the metric")
    n: int = Field(..., description="Total number of trials")
    k: int = Field(..., description="Number of draws")
    c: int = Field(..., description="Number of passing trials")
    pass_at_k: float = Field(..., description="pass@k score: probability at least 1 of k draws passes")
    pass_power_k: float = Field(..., description="pass^k score: probability all k draws pass")
    threshold: float = Field(..., description="Threshold used to determine pass/fail")
    per_trial_scores: list[float] = Field(default_factory=list, description="Individual normalized scores per trial")
    per_trial_passed: list[bool] = Field(default_factory=list, description="Which trials passed the threshold")


class RecordMetrics(BaseModel):
    """All metrics for a single record."""

    model_config = {"extra": "allow"}  # Allow extra fields for backwards compatibility

    record_id: str = Field(..., description="ID of the evaluation record")
    context: dict[str, Any] | None = Field(default=None, description="MetricContext fields used for computing metrics")
    metrics: dict[str, MetricScore] = Field(default_factory=dict, description="Metrics keyed by metric name")
    aggregate_metrics: dict[str, float | None] = Field(
        default_factory=dict,
        description="EVA composite aggregate scores (EVA-A, EVA-X, EVA-overall)",
    )

    def get_score(self, metric_name: str) -> float | None:
        """Get the normalized score for a metric, falling back to raw score."""
        if metric_name not in self.metrics:
            return None
        metric = self.metrics[metric_name]
        if metric.error:
            return None
        return metric.normalized_score if metric.normalized_score is not None else metric.score

    def get_context_field(self, field_name: str) -> Any | None:
        """Safely get a field from context."""
        if self.context and isinstance(self.context, dict):
            return self.context.get(field_name)
        return None


@dataclass
class RunResult:
    """Lightweight result returned by BenchmarkRunner methods (not written to disk)."""

    run_id: str
    total_records: int
    successful_records: int
    failed_records: int
    duration_seconds: float

    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.successful_records / self.total_records
