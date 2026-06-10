"""Tests for ConversationWorker helper methods."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eva.orchestrator.worker import ConversationWorker, _percentile


class TestPercentile:
    def test_single_element_all_percentiles_equal(self):
        assert _percentile([42.0], 1) == 42.0
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 100) == 42.0

    def test_five_elements_p50_is_median(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 50) == 30.0

    def test_100_elements_p95_and_p99(self):
        data = [float(x) for x in range(1, 101)]
        assert _percentile(data, 95) == 95.0
        assert _percentile(data, 99) == 99.0

    def test_nearest_rank_rounds_up(self):
        """ceil(0.33 * 3) = 1 → first element."""
        assert _percentile([10.0, 20.0, 30.0], 33) == 10.0
        # ceil(0.34 * 3) = ceil(1.02) = 2 → second element
        assert _percentile([10.0, 20.0, 30.0], 34) == 20.0


def _make_worker(tmp_path: Path) -> ConversationWorker:
    config = MagicMock()
    config.conversation_time_limit_seconds = 60
    record = MagicMock()
    record.id = "test-record"
    record.current_date_time = "2026-01-01T00:00:00"
    record.user_config = {}
    record.user_goal = "Test goal"

    return ConversationWorker(
        config=config,
        record=record,
        agent=MagicMock(tool_module_path=None),
        agent_config_path="/fake/agents.yaml",
        scenario_base_path="/fake/scenarios",
        output_dir=tmp_path / "output",
        port=9999,
        output_id="test-record",
    )


class TestCalculateLlmLatency:
    def test_no_audit_log_returns_none(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        assert worker._calculate_llm_latency() is None

    def test_empty_llm_prompts_returns_none(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        (worker.output_dir / "audit_log.json").write_text(json.dumps({"llm_prompts": []}))
        assert worker._calculate_llm_latency() is None

    def test_correct_stats_from_five_calls(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        audit = {"llm_prompts": [{"latency_ms": v} for v in [100, 200, 300, 400, 500]]}
        (worker.output_dir / "audit_log.json").write_text(json.dumps(audit))

        result = worker._calculate_llm_latency()
        assert result.total_calls == 5
        assert result.mean_ms == 300.0
        assert result.p50_ms == 300.0  # median of sorted [100,200,300,400,500]
        assert result.p95_ms == 500.0
        assert result.p99_ms == 500.0

    def test_filters_zero_negative_null_and_out_of_range(self, tmp_path):
        """Only valid latencies (0 < ms < 60000) should be included."""
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        audit = {
            "llm_prompts": [
                {"latency_ms": 150},
                {"latency_ms": 0},  # filtered: not > 0
                {"latency_ms": -10},  # filtered: not > 0
                {"latency_ms": 70000},  # filtered: > 60000
                {"latency_ms": None},  # filtered: None
                {"latency_ms": 250},
            ]
        }
        (worker.output_dir / "audit_log.json").write_text(json.dumps(audit))

        result = worker._calculate_llm_latency()
        assert result.total_calls == 2
        assert result.mean_ms == 200.0


class TestCalculateSttLatency:
    def test_no_metrics_file_returns_none(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        assert worker._calculate_stt_latency() is None

    def test_ignores_non_stt_metrics(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        line = json.dumps({"type": "TTFBMetricsData", "processor": "CartesiaTTSService", "value": 0.1})
        (worker.output_dir / "pipecat_metrics.jsonl").write_text(line + "\n")
        assert worker._calculate_stt_latency() is None

    def test_computes_stats_and_converts_to_ms(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        lines = "\n".join(
            [
                json.dumps({"type": "TTFBMetricsData", "processor": "DeepgramSTTService", "value": v})
                for v in [0.1, 0.2, 0.3]
            ]
        )
        (worker.output_dir / "pipecat_metrics.jsonl").write_text(lines + "\n")

        result = worker._calculate_stt_latency()
        assert result.total_calls == 3
        assert result.mean_ms == pytest.approx(200.0, abs=1)
        assert result.p50_ms == pytest.approx(200.0, abs=1)  # median of [100,200,300]

    def test_filters_zero_and_over_30s(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        lines = "\n".join(
            [
                json.dumps({"type": "TTFBMetricsData", "processor": "DeepgramSTTService", "value": v})
                for v in [0.1, 0, 50]  # 0 and 50s filtered
            ]
        )
        (worker.output_dir / "pipecat_metrics.jsonl").write_text(lines + "\n")

        result = worker._calculate_stt_latency()
        assert result.total_calls == 1
        assert result.mean_ms == pytest.approx(100.0, abs=1)


class TestCalculateTtsLatency:
    def test_no_metrics_file_returns_none(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        assert worker._calculate_tts_latency() is None

    def test_computes_tts_ttfb_stats(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        lines = "\n".join(
            [
                json.dumps({"type": "TTFBMetricsData", "processor": "CartesiaTTSService", "value": v})
                for v in [0.05, 0.15]
            ]
        )
        (worker.output_dir / "pipecat_metrics.jsonl").write_text(lines + "\n")

        result = worker._calculate_tts_latency()
        assert result.total_calls == 2
        assert result.mean_ms == pytest.approx(100.0, abs=1)

    def test_filters_over_10s_and_zero(self, tmp_path):
        worker = _make_worker(tmp_path)
        worker.output_dir.mkdir(parents=True)
        lines = "\n".join(
            [
                json.dumps({"type": "TTFBMetricsData", "processor": "CartesiaTTSService", "value": v})
                for v in [0.1, 15, 0]  # 15s and 0 filtered
            ]
        )
        (worker.output_dir / "pipecat_metrics.jsonl").write_text(lines + "\n")

        result = worker._calculate_tts_latency()
        assert result.total_calls == 1


class TestCleanup:
    @pytest.mark.asyncio
    async def test_stops_server_and_clears_references(self, tmp_path):
        worker = _make_worker(tmp_path)
        mock_server = MagicMock()
        mock_server.stop = AsyncMock()
        worker._assistant_server = mock_server
        worker._user_simulator = MagicMock()

        await worker._cleanup()

        mock_server.stop.assert_called_once()
        assert worker._assistant_server is None
        assert worker._user_simulator is None

    @pytest.mark.asyncio
    async def test_server_stop_error_does_not_propagate(self, tmp_path):
        """Cleanup must succeed even if server.stop() raises."""
        worker = _make_worker(tmp_path)
        mock_server = MagicMock()
        mock_server.stop = AsyncMock(side_effect=RuntimeError("socket error"))
        worker._assistant_server = mock_server

        await worker._cleanup()  # Should not raise
        assert worker._assistant_server is None

    @pytest.mark.asyncio
    async def test_cleanup_when_nothing_initialized(self, tmp_path):
        worker = _make_worker(tmp_path)
        await worker._cleanup()  # Should not raise


class TestRunConversation:
    @pytest.mark.asyncio
    async def test_raises_when_simulator_not_initialized(self, tmp_path):
        worker = _make_worker(tmp_path)
        with pytest.raises(RuntimeError, match="User simulator not initialized"):
            await worker._run_conversation()

    @pytest.mark.asyncio
    async def test_returns_ended_reason(self, tmp_path):
        worker = _make_worker(tmp_path)
        mock_sim = MagicMock()
        mock_sim.run_conversation = AsyncMock(return_value="goodbye")
        worker._user_simulator = mock_sim

        result = await worker._run_conversation()

        assert result == "goodbye"


def _setup_run_mocks(worker: ConversationWorker, stats: dict, run_conversation_side_effect=None):
    """Wire up the mocks needed to run worker.run() in isolation.

    Sets worker._assistant_server to a mock that returns *stats* from
    get_conversation_stats(), writes the two DB files run() expects on disk,
    and returns a context manager that patches the expensive internal methods.
    """
    worker.output_dir.mkdir(parents=True, exist_ok=True)
    (worker.output_dir / "initial_scenario_db.json").write_text(json.dumps({}))
    (worker.output_dir / "final_scenario_db.json").write_text(json.dumps({}))

    mock_server = MagicMock()
    mock_server.get_conversation_stats.return_value = stats
    worker._assistant_server = mock_server

    run_conv_mock = AsyncMock(
        side_effect=run_conversation_side_effect,
        return_value="goodbye" if run_conversation_side_effect is None else None,
    )

    patches = [
        patch.object(worker, "_start_assistant", AsyncMock()),
        patch.object(worker, "_start_user_simulator", AsyncMock()),
        patch.object(worker, "_cleanup", AsyncMock()),
        patch.object(worker, "_run_conversation", run_conv_mock),
        patch.object(worker, "_calculate_llm_latency", return_value=None),
        patch.object(worker, "_calculate_stt_latency", return_value=None),
        patch.object(worker, "_calculate_tts_latency", return_value=None),
        patch.object(worker, "_calculate_model_response_latency", return_value=None),
        patch("eva.orchestrator.worker.add_record_log_file", return_value=MagicMock()),
    ]

    class _Ctx:
        async def __aenter__(self_):
            for p in patches:
                p.start()
            return self_

        async def __aexit__(self_, *_args):
            for p in reversed(patches):
                p.stop()

    return _Ctx()


class TestConversationStatsInRun:
    @pytest.mark.asyncio
    async def test_stats_captured_on_normal_completion(self, tmp_path):
        worker = _make_worker(tmp_path)
        stats = {"num_turns": 4, "num_tool_calls": 2, "tools_called": ["lookup_user"]}

        async with _setup_run_mocks(worker, stats):
            result = await worker.run()

        assert result.num_turns == 4
        assert result.num_tool_calls == 2
        assert result.conversation_ended_reason != "error"

    @pytest.mark.asyncio
    async def test_stats_captured_on_time_limit_exceeded(self, tmp_path):
        """Regression test: num_turns must be non-zero even when the conversation times out."""
        worker = _make_worker(tmp_path)
        stats = {"num_turns": 3, "num_tool_calls": 1, "tools_called": ["lookup_user"]}

        async with _setup_run_mocks(worker, stats, run_conversation_side_effect=TimeoutError()):
            result = await worker.run()

        assert result.num_turns == 3
        assert result.num_tool_calls == 1
        assert result.conversation_ended_reason == "time_limit_exceeded"
