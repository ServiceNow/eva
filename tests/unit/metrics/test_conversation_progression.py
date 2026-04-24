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

    def test_build_metric_score_surfaces_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 2,
            "dimensions": {
                "unnecessary_tool_calls": {"rating": 3, "flagged": False, "evidence": "clean"},
                "information_loss": {"rating": 2, "flagged": True, "evidence": "minor"},
                "redundant_statements": {"rating": 3, "flagged": False, "evidence": ""},
                "question_quality": {"rating": 1, "flagged": True, "evidence": "bad"},
            },
        }

        score = self.metric.build_metric_score(
            rating=1,
            normalized=0.0,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response="{...}",
        )

        assert score.sub_metrics is not None
        assert set(score.sub_metrics.keys()) == {
            "unnecessary_tool_calls_rate",
            "information_loss_rate",
            "redundant_statements_rate",
            "question_quality_rate",
        }
        # Binary issue-flag: 1.0 when flagged, 0.0 when clean; lower is better.
        q_quality = score.sub_metrics["question_quality_rate"]
        assert q_quality.name == "conversation_progression.question_quality_rate"
        assert q_quality.score == 1.0  # flagged
        assert q_quality.normalized_score == 1.0
        assert q_quality.details["flagged"] is True
        assert q_quality.details["rating"] == 1
        assert q_quality.details["evidence"] == "bad"

        clean = score.sub_metrics["unnecessary_tool_calls_rate"]
        assert clean.score == 0.0
        assert clean.details["flagged"] is False

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
