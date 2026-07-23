"""conversation_correctly_finished emits per-cause rate sub-metrics on every record.

Cause rates are emitted on all records (0.0 when the conversation finished correctly) so
their cross-record mean reads as "fraction of all conversations with this cause" — matching
how faithfulness dimension rates aggregate. The orthogonal input-characteristic flags
(``final_turn_*``) remain failure-only, since they characterize the unanswered final turn.
"""

import json

import pytest

from eva.metrics.diagnostic.conversation_correctly_finished import ConversationCorrectlyFinishedMetric

from .conftest import make_metric_context


@pytest.fixture
def metric():
    return ConversationCorrectlyFinishedMetric()


def _fail_ctx(tmp_path, **overrides):
    # inactivity_timeout + user last speaker → parent failure (score 0.0)
    return make_metric_context(
        output_dir=str(tmp_path),
        conversation_ended_reason="inactivity_timeout",
        audio_timestamps_user_turns={0: [(0.0, 5.0)]},
        audio_timestamps_assistant_turns={0: [(1.0, 2.0)]},
        **overrides,
    )


@pytest.mark.asyncio
async def test_success_emits_zero_cause_rates(metric):
    from eva.metrics.diagnostic.conv_finish_classifier import CATEGORY_PRIORITY

    ctx = make_metric_context(
        conversation_ended_reason="goodbye",
        audio_timestamps_user_turns={0: [(0.0, 1.0)]},
        audio_timestamps_assistant_turns={0: [(1.5, 3.0)]},
    )
    result = await metric.compute(ctx)
    assert result.score == 1.0
    # Cause rates are present on every record now (denominator = all records, like faithfulness),
    # all 0.0 on a clean finish.
    subs = result.sub_metrics
    assert subs is not None
    assert all(subs[f"{c}_rate"].score == 0.0 for c in CATEGORY_PRIORITY)
    assert sum(subs[f"{c}_rate"].score for c in CATEGORY_PRIORITY) == 0.0
    # Input-characteristic flags remain failure-only.
    assert not any(k.startswith("final_turn_") for k in subs)


@pytest.mark.asyncio
async def test_failure_reasoning_only_sub_metric_fires(metric, tmp_path):
    (tmp_path / "agent_perf_stats.csv").write_text('response,tool_calls,reasoning\n,,"Booked room 201."\n')
    (tmp_path / "audit_log.json").write_text(
        json.dumps({"conversation_messages": [{"role": "assistant", "content": ""}]})
    )
    result = await metric.compute(_fail_ctx(tmp_path))
    assert result.score == 0.0
    subs = result.sub_metrics
    assert subs["reasoning_only_rate"].score == 1.0
    assert subs["stt_empty_transcription_rate"].score == 0.0
    assert subs["unknown_reason_rate"].score == 0.0
    # exactly one *cause* flag is hot (input-characteristic flags are orthogonal, excluded here)
    from eva.metrics.diagnostic.conv_finish_classifier import CATEGORY_PRIORITY

    assert sum(subs[f"{c}_rate"].score for c in CATEGORY_PRIORITY) == 1.0
    # fired sub-metric name is namespaced under the parent
    assert subs["reasoning_only_rate"].name == "conversation_correctly_finished.reasoning_only_rate"


@pytest.mark.asyncio
async def test_failure_with_no_files_is_unknown(metric, tmp_path):
    (tmp_path / "audit_log.json").write_text(
        json.dumps({"conversation_messages": [{"role": "assistant", "content": "x"}]})
    )
    result = await metric.compute(_fail_ctx(tmp_path))
    assert result.score == 0.0
    assert result.sub_metrics["unknown_reason_rate"].score == 1.0


def _fixture_with_final_user_speech(tmp_path, text):
    (tmp_path / "audit_log.json").write_text(
        json.dumps({"conversation_messages": [{"role": "assistant", "content": "x"}]})
    )
    (tmp_path / "user_simulator_events.jsonl").write_text(json.dumps({"type": "user_speech", "data": {"text": text}}))
    return tmp_path


@pytest.mark.asyncio
async def test_acknowledgement_final_turn_flag(metric, tmp_path):
    result = await metric.compute(_fail_ctx(_fixture_with_final_user_speech(tmp_path, "Yes, that is correct.")))
    subs = result.sub_metrics
    assert subs["final_turn_acknowledgement_rate"].score == 1.0
    assert subs["final_turn_short_rate"].score == 0.0  # 4 words


@pytest.mark.asyncio
async def test_short_final_turn_flag(metric, tmp_path):
    subs = (await metric.compute(_fail_ctx(_fixture_with_final_user_speech(tmp_path, "Yes.")))).sub_metrics
    assert subs["final_turn_short_rate"].score == 1.0
    assert subs["final_turn_acknowledgement_rate"].score == 1.0


@pytest.mark.asyncio
async def test_spelled_entity_final_turn_flag(metric, tmp_path):
    fx = _fixture_with_final_user_speech(tmp_path, "It is D I A G dash two X M nine five P L Q.")
    subs = (await metric.compute(_fail_ctx(fx))).sub_metrics
    assert subs["final_turn_spelled_entity_rate"].score == 1.0
    assert subs["final_turn_acknowledgement_rate"].score == 0.0


@pytest.mark.asyncio
async def test_unknown_reason_fires_even_with_input_flags_present(metric, tmp_path):
    # An input flag being set must NOT suppress unknown_reason — the flags are orthogonal to causes.
    subs = (
        await metric.compute(_fail_ctx(_fixture_with_final_user_speech(tmp_path, "Yes, that is correct.")))
    ).sub_metrics
    assert subs["final_turn_acknowledgement_rate"].score == 1.0  # input flag present
    assert subs["unknown_reason_rate"].score == 1.0  # …and the record is still tagged unknown


@pytest.mark.asyncio
async def test_input_flags_absent_on_success(metric):
    ctx = make_metric_context(
        conversation_ended_reason="goodbye",
        audio_timestamps_user_turns={0: [(0.0, 1.0)]},
        audio_timestamps_assistant_turns={0: [(1.5, 3.0)]},
    )
    result = await metric.compute(ctx)
    # Input-characteristic flags are only emitted on failures; cause rates are still present (all 0.0).
    assert not any(k.startswith("final_turn_") for k in result.sub_metrics)
