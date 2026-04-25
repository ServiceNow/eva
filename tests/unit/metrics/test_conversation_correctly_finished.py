"""Tests for ConversationCorrectlyFinishedMetric."""

import pytest

from eva.metrics.diagnostic.conversation_correctly_finished import ConversationCorrectlyFinishedMetric

from .conftest import make_metric_context


@pytest.fixture
def metric():
    return ConversationCorrectlyFinishedMetric()


@pytest.mark.asyncio
async def test_goodbye_scores_1(metric):
    context = make_metric_context(
        conversation_ended_reason="goodbye",
        audio_timestamps_user_turns={0: [(0.0, 1.0)]},
        audio_timestamps_assistant_turns={0: [(1.5, 3.0)]},
    )
    result = await metric.compute(context)
    assert result.score == 1.0
    assert result.normalized_score == 1.0
    assert result.details["conversation_ended_reason"] == "goodbye"
    assert result.details["last_audio_speaker"] == "assistant"


@pytest.mark.asyncio
async def test_error_scores_1(metric):
    context = make_metric_context(
        conversation_ended_reason="error",
        audio_timestamps_user_turns={0: [(0.0, 5.0)]},
        audio_timestamps_assistant_turns={0: [(1.0, 2.0)]},
    )
    result = await metric.compute(context)
    assert result.score == 1.0
    assert result.details["conversation_ended_reason"] == "error"


@pytest.mark.asyncio
async def test_inactivity_user_last_scores_0(metric):
    context = make_metric_context(
        conversation_ended_reason="inactivity_timeout",
        audio_timestamps_user_turns={0: [(0.0, 5.0)]},
        audio_timestamps_assistant_turns={0: [(1.0, 2.0)]},
    )
    result = await metric.compute(context)
    assert result.score == 0.0
    assert result.normalized_score == 0.0
    assert result.details["conversation_ended_reason"] == "inactivity_timeout"
    assert result.details["last_audio_speaker"] == "user"
    assert "user was the last speaker" in result.details["reason"]


@pytest.mark.asyncio
async def test_inactivity_assistant_last_scores_1(metric):
    context = make_metric_context(
        conversation_ended_reason="inactivity_timeout",
        audio_timestamps_user_turns={0: [(0.0, 2.0)]},
        audio_timestamps_assistant_turns={0: [(1.0, 5.0)]},
    )
    result = await metric.compute(context)
    assert result.score == 1.0
    assert result.details["last_audio_speaker"] == "assistant"


@pytest.mark.asyncio
async def test_inactivity_no_audio_scores_1(metric):
    context = make_metric_context(
        conversation_ended_reason="inactivity_timeout",
        audio_timestamps_user_turns={},
        audio_timestamps_assistant_turns={},
    )
    result = await metric.compute(context)
    assert result.score == 1.0
    assert result.details["last_audio_speaker"] is None
