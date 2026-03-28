"""Tests for ConversationWorker helper methods."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.models.config import TelephonyBridgeConfig
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
    config.conversation_timeout_seconds = 60
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
                json.dumps({"type": "ProcessingMetricsData", "processor": "DeepgramSTTService", "value": v})
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
                json.dumps({"type": "ProcessingMetricsData", "processor": "DeepgramSTTService", "value": v})
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

    @pytest.mark.asyncio
    async def test_unregisters_tool_webhook_after_stopping_telephony_server(self, tmp_path):
        worker = _make_worker(tmp_path)
        mock_server = MagicMock()
        mock_server.stop = AsyncMock()
        mock_webhook = MagicMock()
        mock_webhook.unregister_conversation = AsyncMock()
        worker._assistant_server = mock_server
        worker.tool_webhook_service = mock_webhook
        worker._registered_tool_call_id = "test-record"

        await worker._cleanup()

        mock_server.stop.assert_called_once()
        mock_webhook.unregister_conversation.assert_awaited_once_with("test-record")
        assert worker._registered_tool_call_id is None


class TestRunConversation:
    @pytest.mark.asyncio
    async def test_raises_when_simulator_not_initialized(self, tmp_path):
        worker = _make_worker(tmp_path)
        with pytest.raises(RuntimeError, match="User simulator not initialized"):
            await worker._run_conversation()

    @pytest.mark.asyncio
    async def test_returns_ended_reason_and_captures_stats(self, tmp_path):
        worker = _make_worker(tmp_path)
        mock_sim = MagicMock()
        mock_sim.run_conversation = AsyncMock(return_value="goodbye")
        worker._user_simulator = mock_sim

        stats = {"num_turns": 5, "num_tool_calls": 2, "tools_called": ["get_reservation"]}
        mock_server = MagicMock()
        mock_server.get_conversation_stats.return_value = stats
        worker._assistant_server = mock_server

        result = await worker._run_conversation()

        assert result == "goodbye"
        assert worker._conversation_stats == stats


class TestTelephonyStart:
    @pytest.mark.asyncio
    async def test_start_assistant_uses_telephony_bridge_and_registers_webhook(self, tmp_path, monkeypatch):
        worker = _make_worker(tmp_path)
        worker.config.model = TelephonyBridgeConfig(
            sip_uri="sip:test@example.com",
            telnyx_api_key="telnyx-key",
            call_control_app_id="app-123",
            call_control_from="+15551234567",
            webhook_base_url="https://example.ngrok-free.dev",
        )
        worker.tool_webhook_service = MagicMock()
        worker.tool_webhook_service.register_conversation = AsyncMock()

        mock_bridge = MagicMock()
        mock_bridge.start = AsyncMock()
        mock_bridge.tool_handler = MagicMock()
        mock_bridge.audit_log = MagicMock()

        bridge_ctor = MagicMock(return_value=mock_bridge)
        monkeypatch.setattr("eva.orchestrator.worker.TelephonyBridgeServer", bridge_ctor)

        await worker._start_assistant()

        bridge_ctor.assert_called_once()
        assert bridge_ctor.call_args.kwargs["telnyx_conversation_lookup"] == (
            worker.tool_webhook_service.get_telnyx_conversation_id
        )
        mock_bridge.start.assert_awaited_once()
        worker.tool_webhook_service.register_conversation.assert_awaited_once_with(
            "test-record",
            mock_bridge.tool_handler,
            audit_log=mock_bridge.audit_log,
        )
        assert worker._registered_tool_call_id == "test-record"
