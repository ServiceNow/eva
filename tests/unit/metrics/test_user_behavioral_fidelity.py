"""Tests for UserBehavioralFidelityMetric."""

import json

import pytest

from eva.metrics.validation.user_behavioral_fidelity import UserBehavioralFidelityMetric
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestUserBehavioralFidelity:
    def setup_method(self):
        self.metric = make_judge_metric(UserBehavioralFidelityMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "user_behavioral_fidelity"
        assert self.metric.category == "validation"
        assert self.metric.rating_scale == (0, 1)

    def test_get_prompt_variables_cascade(self):
        ctx = make_metric_context(
            user_goal="Book a flight",
            user_persona="Friendly traveler",
            agent_tools=[
                {"name": "search_flights", "tool_type": "read"},
                {"name": "book_flight", "tool_type": "write"},
            ],
            pipeline_type="cascade",
            intended_user_turns={0: "Hi, I need to book a flight"},
        )
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")

        assert variables["user_goal"] == "Book a flight"
        assert variables["user_persona"] == "Friendly traveler"
        # Only write tools should be in modification_tools
        mod_tools = json.loads(variables["modification_tools"])
        assert len(mod_tools) == 1
        assert mod_tools[0]["name"] == "book_flight"
        # Cascade evidence should include agent-side transcript label
        assert "Agent-Side Transcript" in variables["conversation_evidence"]

    def test_get_prompt_variables_s2s(self):
        ctx = make_metric_context(
            user_goal="Book a flight",
            user_persona="Friendly traveler",
            agent_tools=[],
            pipeline_type="s2s",
            intended_user_turns={0: "Hi"},
        )
        variables = self.metric.get_prompt_variables(ctx, "transcript text")
        assert "speech-to-speech" in variables["conversation_evidence"]

    def test_build_metric_score_not_corrupted(self):
        ctx = make_metric_context()
        response = {"corruption_analysis": {"goal_drift": False}}

        score = self.metric.build_metric_score(
            rating=1,
            normalized=1.0,
            response=response,
            prompt="test",
            context=ctx,
        )

        assert score.score == 1.0
        assert score.details["corrupted"] is False
        assert score.details["corruption_analysis"] == {"goal_drift": False}

    def test_build_metric_score_corrupted(self):
        ctx = make_metric_context()
        response = {"corruption_analysis": {"goal_drift": True}}

        score = self.metric.build_metric_score(
            rating=0,
            normalized=0.0,
            response=response,
            prompt="test",
            context=ctx,
        )

        assert score.score == 0.0
        assert score.details["corrupted"] is True

    @pytest.mark.asyncio
    async def test_compute_not_corrupted(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 1,
                "corruption_analysis": {},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "book flight"},
                {"role": "assistant", "content": "sure"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_compute_corrupted(self):
        self.metric.llm_client.generate_text.return_value = json.dumps(
            {
                "rating": 0,
                "corruption_analysis": {"goal_drift": True},
            }
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "actually never mind"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.normalized_score == 0.0
