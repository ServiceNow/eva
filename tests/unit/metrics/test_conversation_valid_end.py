"""Tests for ConversationValidEndMetric."""

import json

import pytest

from eva.metrics.validation.conversation_valid_end import ConversationValidEndMetric
from tests.unit.metrics.conftest import make_metric_context


class TestConversationValidEnd:
    def setup_method(self):
        self.metric = ConversationValidEndMetric()

    def _write_events(self, tmp_path, events: list[dict]) -> str:
        events_file = tmp_path / "elevenlabs_events.jsonl"
        with open(events_file, "w") as f:
            f.writelines(json.dumps(event) + "\n" for event in events)
        return str(tmp_path)

    @pytest.mark.asyncio
    async def test_properly_ended_conversation(self, tmp_path):
        events = [
            {"type": "audio", "data": {}},
            {"type": "connection_state", "data": {"details": {"reason": "goodbye"}}},
        ]
        output_dir = self._write_events(tmp_path, events)
        ctx = make_metric_context(output_dir=output_dir)

        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.details["ended_properly"] is True

    @pytest.mark.asyncio
    async def test_missing_events_file(self, tmp_path):
        ctx = make_metric_context(output_dir=str(tmp_path))
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert "not found" in score.error

    @pytest.mark.asyncio
    async def test_empty_events_file(self, tmp_path):
        (tmp_path / "elevenlabs_events.jsonl").write_text("")
        ctx = make_metric_context(output_dir=str(tmp_path))
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert "empty" in score.error

    @pytest.mark.asyncio
    async def test_last_event_wrong_type(self, tmp_path):
        events = [{"type": "audio", "data": {}}]
        output_dir = self._write_events(tmp_path, events)
        ctx = make_metric_context(output_dir=output_dir)

        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.details["last_event_type"] == "audio"

    @pytest.mark.asyncio
    async def test_connection_state_wrong_reason(self, tmp_path):
        events = [
            {"type": "connection_state", "data": {"details": {"reason": "timeout"}}},
        ]
        output_dir = self._write_events(tmp_path, events)
        ctx = make_metric_context(output_dir=output_dir)

        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.details["ended_properly"] is False

    @pytest.mark.asyncio
    async def test_malformed_json_last_line(self, tmp_path):
        events_file = tmp_path / "elevenlabs_events.jsonl"
        events_file.write_text("not valid json\n")
        ctx = make_metric_context(output_dir=str(tmp_path))

        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert "Failed to parse" in score.error

    def test_metric_attributes(self):
        assert self.metric.name == "conversation_valid_end"
        assert self.metric.category == "validation"
