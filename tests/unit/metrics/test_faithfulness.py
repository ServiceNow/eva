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

    def test_build_metric_score_surfaces_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 2,
            "dimensions": {
                "fabricating_tool_parameters": {"rating": 3, "flagged": False, "evidence": "clean"},
                "misrepresenting_tool_result": {"rating": 2, "flagged": True, "evidence": "minor"},
                "violating_policies": {"rating": 3, "flagged": False, "evidence": ""},
                "failing_to_disambiguate": {"rating": 1, "flagged": True, "evidence": "bad"},
                "hallucination": {"rating": 3, "flagged": False, "evidence": ""},
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
            "fabricating_tool_parameters_rate",
            "misrepresenting_tool_result_rate",
            "violating_policies_rate",
            "failing_to_disambiguate_rate",
            "hallucination_rate",
        }
        # Binary issue-flag semantics: 1.0 when flagged, 0.0 when clean.
        fab = score.sub_metrics["fabricating_tool_parameters_rate"]
        assert fab.name == "faithfulness.fabricating_tool_parameters_rate"
        assert fab.score == 0.0  # clean
        assert fab.normalized_score == 0.0
        assert fab.details["flagged"] is False
        assert fab.details["rating"] == 3  # raw rating preserved for diagnostics

        disamb = score.sub_metrics["failing_to_disambiguate_rate"]
        assert disamb.score == 1.0  # flagged
        assert disamb.normalized_score == 1.0
        assert disamb.details["flagged"] is True
        assert disamb.details["rating"] == 1
        assert disamb.details["evidence"] == "bad"

    def test_build_metric_score_skips_dimensions_without_flag(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}])
        response = {
            "rating": 3,
            "dimensions": {
                "fabricating_tool_parameters": {"rating": 3},  # no flagged field
                "hallucination": {"rating": 3, "flagged": False},
            },
        }

        score = self.metric.build_metric_score(
            rating=3,
            normalized=1.0,
            response=response,
            prompt="p",
            context=ctx,
            raw_response="{}",
        )

        assert score.sub_metrics is not None
        assert set(score.sub_metrics.keys()) == {"hallucination_rate"}

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
