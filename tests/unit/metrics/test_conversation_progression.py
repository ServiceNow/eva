"""Tests for ConversationProgressionJudgeMetric."""

import json

import pytest

from eva.metrics.experience.conversation_progression import ConversationProgressionJudgeMetric
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestConversationProgression:
    def setup_method(self):
        self.metric = make_judge_metric(ConversationProgressionJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "conversation_progression"
        assert self.metric.category == "experience"
        assert self.metric.rating_scale == (1, 3)

    def test_get_prompt_variables(self):
        ctx = make_metric_context()
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["conversation_trace"] == "User: hi\nBot: hello"

    def test_build_metric_score(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}, {"role": "user"}])
        response = {"dimensions": {"progress": "good"}, "flags_count": 0}

        score = self.metric.build_metric_score(
            rating=2,
            normalized=0.5,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response='{"rating": 2}',
        )

        assert score.name == "conversation_progression"
        assert score.score == 2.0
        assert score.normalized_score == 0.5
        assert score.details["explanation"]["dimensions"] == {"progress": "good"}
        assert score.details["explanation"]["flags_count"] == 0
        assert score.details["num_turns"] == 3

    @pytest.mark.asyncio
    async def test_compute_uses_message_trace_for_continuous_stream(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 3,
                "dimensions": {},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "book flight", "turn_id": 1},
            ],
            message_trace=[
                {"role": "assistant", "content": "Welcome to EVA.", "turn_id": 0},
                {"role": "user", "content": "book flight", "turn_id": 1},
                {"role": "assistant", "content": "I found your reservation.", "turn_id": 1},
            ],
            is_continuous_assistant_stream=True,
        )

        score = await self.metric.compute(ctx)

        prompt = self.metric.llm_client.generate_text.await_args.args[0][0]["content"]
        assert "I found your reservation." in prompt
        assert score.details["num_turns"] == 3

    @pytest.mark.asyncio
    async def test_compute_keeps_validated_trace_for_non_continuous_stream(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 3,
                "dimensions": {},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "book flight", "turn_id": 1},
                {"role": "assistant", "content": "Validated assistant text.", "turn_id": 1},
            ],
            message_trace=[
                {"role": "user", "content": "book flight", "turn_id": 1},
                {"role": "assistant", "content": "Unvalidated native assistant text.", "turn_id": 1},
            ],
            is_continuous_assistant_stream=False,
        )

        score = await self.metric.compute(ctx)

        prompt = self.metric.llm_client.generate_text.await_args.args[0][0]["content"]
        assert "Validated assistant text." in prompt
        assert "Unvalidated native assistant text." not in prompt
        assert score.details["num_turns"] == 2

    @pytest.mark.asyncio
    async def test_compute_excellent(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 3,
                "dimensions": {},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "book flight"},
                {"role": "assistant", "content": "done"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_compute_poor(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 1,
                "dimensions": {},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "help"},
                {"role": "assistant", "content": "sorry"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 0.0
