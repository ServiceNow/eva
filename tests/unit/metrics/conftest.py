"""Shared fixtures for metric tests."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from eva.metrics.base import MetricContext
from eva.metrics.processor import MetricsContextProcessor
from eva.models.results import MetricScore, RecordMetrics


def make_metric_context(**overrides) -> MetricContext:
    """Create a MetricContext with sensible defaults, overridable via kwargs.

    If audio_timestamps_*_turns are provided but latency_assistant_turns is not,
    it is automatically derived from the timestamps.

    Usage::

        ctx = make_metric_context(conversation_trace=[...])
        ctx = make_metric_context(tool_responses=[...], tool_params=[...])
    """
    defaults = {
        "record_id": "test_record",
        "user_goal": "Test goal",
        "user_persona": "Test persona",
        "expected_scenario_db": {},
        "initial_scenario_db": {},
        "final_scenario_db": {},
        "initial_scenario_db_hash": "",
        "final_scenario_db_hash": "",
        "agent_role": "Test role",
        "agent_instructions": "Test instructions",
        "agent_tools": [],
        "current_date_time": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)

    # Auto-derive latency_assistant_turns from audio timestamps if not provided
    if "latency_assistant_turns" not in overrides:
        user_ts = defaults.get("audio_timestamps_user_turns")
        asst_ts = defaults.get("audio_timestamps_assistant_turns")
        if user_ts and asst_ts:
            tmp = SimpleNamespace(
                audio_timestamps_user_turns=user_ts,
                audio_timestamps_assistant_turns=asst_ts,
                latency_assistant_turns={},
            )
            MetricsContextProcessor._compute_per_turn_latency(tmp)
            defaults["latency_assistant_turns"] = tmp.latency_assistant_turns

    return MetricContext(**defaults)


def make_judge_metric(metric_cls, *, mock_llm: bool = False, logger_name: str | None = None):
    """Instantiate a judge metric, loading real prompts from configs/prompts/.

    Args:
        metric_cls: The metric class to instantiate.
        mock_llm: If True, also replace ``llm_client`` with a MagicMock.
        logger_name: If provided, replace the metric's logger.

    Returns:
        An instance of *metric_cls* ready for testing.
    """
    m = metric_cls()
    if logger_name:
        m.logger = logging.getLogger(logger_name)
    if mock_llm:
        m.llm_client = MagicMock()
        m.llm_client.generate_text = AsyncMock()
    return m


def make_metric_score(
    name: str,
    score: float = 1.0,
    normalized_score: float | None = None,
    error: str | None = None,
    details: dict | None = None,
) -> MetricScore:
    """Shorthand to build a MetricScore."""
    return MetricScore(
        name=name,
        score=score,
        normalized_score=normalized_score if normalized_score is not None else score,
        error=error,
        details=details or {},
    )


def make_record_metrics(scores: dict[str, float], record_id: str = "1.1.1") -> RecordMetrics:
    """Create RecordMetrics from a {name: normalized_score} dict."""
    metrics = {}
    for name, value in scores.items():
        metrics[name] = MetricScore(name=name, score=value, normalized_score=value)
    return RecordMetrics(record_id=record_id, metrics=metrics)
