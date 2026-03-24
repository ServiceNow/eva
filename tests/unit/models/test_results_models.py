"""Unit tests for results models."""

from datetime import datetime

from eva.models.results import (
    ConversationResult,
    MetricScore,
    RecordMetrics,
    RunResult,
)


class TestConversationResult:
    def test_create_successful_result(self):
        """Test creating a successful conversation result."""
        now = datetime.now()
        result = ConversationResult(
            record_id="test_001",
            completed=True,
            started_at=now,
            ended_at=now,
            duration_seconds=45.5,
            output_dir="/output/test_001",
            num_turns=5,
            num_tool_calls=2,
            tools_called=["tool_a", "tool_b"],
            conversation_ended_reason="goodbye",
        )

        assert result.record_id == "test_001"
        assert result.completed is True
        assert result.error is None
        assert result.num_turns == 5
        assert result.tools_called == ["tool_a", "tool_b"]
        assert result.conversation_ended_reason == "goodbye"

    def test_create_failed_result(self):
        """Test creating a failed conversation result."""
        now = datetime.now()
        result = ConversationResult(
            record_id="test_002",
            completed=False,
            error="Connection timeout",
            started_at=now,
            ended_at=now,
            duration_seconds=120.0,
            output_dir="/output/test_002",
            conversation_ended_reason="error",
        )

        assert result.completed is False
        assert result.error == "Connection timeout"
        assert result.conversation_ended_reason == "error"

    def test_default_values(self):
        """Test default values for ConversationResult."""
        now = datetime.now()
        result = ConversationResult(
            record_id="test_003",
            completed=True,
            started_at=now,
            ended_at=now,
            duration_seconds=30.0,
            output_dir="/output/test_003",
        )

        assert result.num_turns == 0
        assert result.num_tool_calls == 0
        assert result.tools_called == []
        assert result.audio_assistant_path is None


class TestMetricScore:
    def test_create_metric_score(self):
        """Test creating a metric score."""
        score = MetricScore(
            name="task_completion",
            score=0.85,
            normalized_score=0.85,
            details={"sub_task_1": 1.0, "sub_task_2": 0.7},
        )

        assert score.name == "task_completion"
        assert score.score == 0.85
        assert score.normalized_score == 0.85
        assert score.error is None

    def test_create_failed_metric(self):
        """Test creating a failed metric score."""
        score = MetricScore(
            name="audio_quality",
            score=0.0,
            error="Audio file not found",
        )

        assert score.error == "Audio file not found"

    def test_default_values(self):
        """Test default values for MetricScore."""
        score = MetricScore(name="test", score=0.5)

        assert score.normalized_score is None
        assert score.details == {}
        assert score.error is None


class TestRecordMetrics:
    def test_create_record_metrics(self):
        """Test creating record metrics."""
        metrics = RecordMetrics(
            record_id="test_001",
            metrics={
                "task_completion": MetricScore(name="task_completion", score=0.9),
                "helpfulness": MetricScore(name="helpfulness", score=0.8),
            },
        )

        assert metrics.record_id == "test_001"
        assert len(metrics.metrics) == 2

    def test_get_score(self):
        """Test getting a score by metric name."""
        metrics = RecordMetrics(
            record_id="test_001",
            metrics={
                "task_completion": MetricScore(
                    name="task_completion",
                    score=0.9,
                    normalized_score=0.95,
                ),
                "failed_metric": MetricScore(
                    name="failed_metric",
                    score=0.0,
                    error="Failed to compute",
                ),
            },
        )

        # Should return normalized score when available
        score = metrics.get_score("task_completion")
        assert score == 0.95

        # Should return None for failed metric
        score = metrics.get_score("failed_metric")
        assert score is None

        # Should return None for nonexistent metric
        score = metrics.get_score("nonexistent")
        assert score is None

    def test_get_score_fallback_to_raw(self):
        """Test that get_score falls back to raw score."""
        metrics = RecordMetrics(
            record_id="test_001",
            metrics={
                "task_completion": MetricScore(
                    name="accuracy",
                    score=0.9,
                    # No normalized_score
                ),
            },
        )

        score = metrics.get_score("task_completion")
        assert score == 0.9


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_create_run_result(self):
        result = RunResult(
            run_id="run_20240115", total_records=100, successful_records=95, failed_records=5, duration_seconds=3600.0
        )
        assert result.run_id == "run_20240115"
        assert result.total_records == 100
        assert result.successful_records == 95

    def test_success_rate(self):
        assert (
            RunResult(
                run_id="t", total_records=100, successful_records=80, failed_records=20, duration_seconds=0.0
            ).success_rate
            == 0.8
        )
        assert (
            RunResult(
                run_id="t", total_records=0, successful_records=0, failed_records=0, duration_seconds=0.0
            ).success_rate
            == 0.0
        )
