"""Tests for eva.metrics.utils module."""

from unittest.mock import MagicMock

import jiwer
import pytest

from eva.metrics.utils import (
    aggregate_per_turn_scores,
    aggregate_wer_errors,
    compute_aggregation,
    extract_wer_errors,
    format_transcript,
    normalize_rating,
    parse_judge_response,
    parse_judge_response_list,
    resolve_turn_id,
    reverse_word_error_rate,
    smart_harmonic_mean,
    validate_rating,
)


class TestParseJudgeResponse:
    def test_valid_json(self):
        result = parse_judge_response('{"rating": 3}', "rec-1", MagicMock())
        assert result == {"rating": 3}

    def test_json_in_text(self):
        result = parse_judge_response('Here is result: {"rating": 2}', "rec-1", MagicMock())
        assert result == {"rating": 2}

    def test_invalid_json_returns_none(self):
        mock_logger = MagicMock()
        result = parse_judge_response("no json here", "rec-1", mock_logger)
        assert result is None
        mock_logger.error.assert_called()


class TestParseJudgeResponseList:
    def test_none_input(self):
        assert parse_judge_response_list(None) is None

    def test_json_array(self):
        result = parse_judge_response_list('[{"turn_id": 1}, {"turn_id": 2}]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_single_dict_wrapped_in_list(self):
        result = parse_judge_response_list('{"turn_id": 1}')
        assert result == [{"turn_id": 1}]

    def test_unparseable_returns_none(self):
        assert parse_judge_response_list("not json") is None


class TestFormatTranscript:
    def test_normal_turns(self):
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_transcript(turns)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result

    def test_empty_list(self):
        assert format_transcript([]) == "No transcript available"

    def test_empty_content_skipped(self):
        turns = [{"role": "user", "content": ""}, {"role": "assistant", "content": "hi"}]
        result = format_transcript(turns)
        assert "User:" not in result
        assert "Assistant: hi" in result


class TestResolveTurnId:
    def test_valid_turn_id(self):
        result = resolve_turn_id({"turn_id": 2, "rating": 3}, [1, 2, 3])
        assert result == 2

    def test_missing_turn_id(self):
        result = resolve_turn_id({"rating": 3}, [1, 2, 3])
        assert result is None

    def test_turn_id_not_in_expected(self):
        result = resolve_turn_id({"turn_id": 99}, [1, 2, 3], "test_metric")
        assert result is None

    def test_non_dict_input(self):
        result = resolve_turn_id("not a dict", [1, 2])
        assert result is None


class TestValidateRating:
    def test_valid_rating(self):
        result = validate_rating(3, [1, 2, 3], 2, "rec-1", MagicMock())
        assert result == 3

    def test_invalid_rating_uses_default(self):
        mock_logger = MagicMock()
        result = validate_rating(5, [1, 2, 3], 2, "rec-1", mock_logger)
        assert result == 2
        mock_logger.warning.assert_called()


class TestNormalizeRating:
    def test_min_value(self):
        assert normalize_rating(1, 1, 3) == pytest.approx(0.0)

    def test_max_value(self):
        assert normalize_rating(3, 1, 3) == pytest.approx(1.0)

    def test_mid_value(self):
        assert normalize_rating(2, 1, 3) == pytest.approx(0.5)

    def test_equal_min_max(self):
        assert normalize_rating(5, 5, 5) == 1.0


class TestReverseWordErrorRate:
    def test_zero_wer(self):
        assert reverse_word_error_rate(0.0) == 1.0

    def test_full_wer(self):
        assert reverse_word_error_rate(1.0) == 0.0

    def test_typical_wer(self):
        assert reverse_word_error_rate(0.3) == pytest.approx(0.7)

    def test_wer_above_one_clamped(self):
        assert reverse_word_error_rate(1.5) == 0.0


class TestSmartHarmonicMean:
    def test_valid_scores(self):
        result = smart_harmonic_mean([1.0, 1.0, 1.0])
        assert result == 1.0

    def test_empty_list(self):
        assert smart_harmonic_mean([]) is None

    def test_with_none_values(self):
        result = smart_harmonic_mean([1.0, None, 1.0])
        assert result == 1.0

    def test_all_none(self):
        assert smart_harmonic_mean([None, None]) is None

    def test_rounded(self):
        result = smart_harmonic_mean([0.5, 0.8])
        assert result == round(result, 3)


class TestComputeAggregation:
    def test_hmean(self):
        result = compute_aggregation("hmean", [1.0, 1.0])
        assert result == 1.0

    def test_mean(self):
        result = compute_aggregation("mean", [0.5, 1.0])
        assert result == pytest.approx(0.75)

    def test_abs_mean(self):
        result = compute_aggregation("abs_mean", [-1.0, 1.0])
        assert result == pytest.approx(1.0)

    def test_min(self):
        result = compute_aggregation("min", [0.3, 0.7, 0.5])
        assert result == 0.3

    def test_empty_list(self):
        assert compute_aggregation("mean", []) is None

    def test_filters_none(self):
        result = compute_aggregation("mean", [None, 0.5, None, 1.0])
        assert result == pytest.approx(0.75)

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            compute_aggregation("invalid", [1.0])


class TestAggregatePerTurnScores:
    def test_delegates_to_compute_aggregation(self):
        result = aggregate_per_turn_scores([0.5, 1.0], "mean")
        assert result == pytest.approx(0.75)


class TestExtractWerErrors:
    def test_extracts_errors(self):
        output = jiwer.process_words("the cat sat", "a cat sit")
        errors = extract_wer_errors(output)
        assert "substitutions" in errors
        assert "deletions" in errors
        assert "insertions" in errors
        # "the"→"a" and "sat"→"sit" are substitutions
        assert len(errors["substitutions"]) == 2


class TestAggregateWerErrors:
    def test_aggregates(self):
        output = jiwer.process_words("the cat sat", "a cat sit")
        result = aggregate_wer_errors(output)
        assert "top_substitutions" in result
        assert "top_deletions" in result
        assert "top_insertions" in result
