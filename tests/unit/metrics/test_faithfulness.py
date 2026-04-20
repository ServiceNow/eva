"""Tests for FaithfulnessJudgeMetric."""

import json

import pytest

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestFaithfulness:
    def setup_method(self):
        self.metric = make_judge_metric(FaithfulnessJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "faithfulness"
        assert self.metric.category == "accuracy"
        assert self.metric.rating_scale == (1, 3)

    def test_get_prompt_variables_cascade(self):
        ctx = make_metric_context(
            agent_instructions="Be helpful",
            agent_role="Assistant",
            agent_tools=[{"name": "search"}],
            current_date_time="2026-01-01",
            pipeline_type="cascade",
        )
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["agent_instructions"] == "Be helpful"
        assert variables["agent_role"] == "Assistant"
        assert "conversation_trace" in variables
        assert "STT" in variables["user_turns_disclaimer"]  # cascade mode
        assert "speech-to-text" in variables["disambiguation_context"]

    def test_get_prompt_variables_s2s(self):
        ctx = make_metric_context(pipeline_type="s2s")
        variables = self.metric.get_prompt_variables(ctx, "transcript")
        assert "speech-to-speech" in variables["user_turns_disclaimer"]
        assert "raw audio" in variables["disambiguation_context"]

    def test_build_metric_score(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {"dimensions": {"hallucination": "none"}}

        score = self.metric.build_metric_score(
            rating=3,
            normalized=1.0,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response='{"rating": 3}',
        )

        assert score.name == "faithfulness"
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.details["rating"] == 3
        assert score.details["explanation"]["dimensions"] == {"hallucination": "none"}
        assert score.details["num_turns"] == 2

    @pytest.mark.asyncio
    async def test_compute_success(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 3,
                "dimensions": {"hallucination": "none"},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_compute_empty_transcript(self):
        ctx = make_metric_context(conversation_trace=[])
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert "No transcript" in score.error

    @pytest.mark.asyncio
    async def test_compute_unparseable_response(self):
        self.metric.llm_client.generate_text.return_value = "not json at all ~~~"
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.error is not None
