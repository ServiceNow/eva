"""Tests for eva.metrics.debug.stt_wer compute method."""

import pytest

from eva.metrics.diagnostic.stt_wer import STTWERMetric

from .conftest import make_metric_context


class TestSTTWERMetricCompute:
    @pytest.mark.asyncio
    async def test_perfect_transcription(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.details["wer"] == 0.0
        assert result.details["accuracy"] == 1.0
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_some_errors(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "the quick brown fox"},
            transcribed_user_turns={1: "the quick green fox"},
        )
        result = await metric.compute(ctx)
        assert result.details["wer"] > 0
        assert result.details["accuracy"] < 1.0
        assert result.normalized_score == result.details["accuracy"]

    @pytest.mark.asyncio
    async def test_surfaces_component_sub_metrics(self):
        """Sub-metrics surface substitution, deletion, insertion rates over reference words."""
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "the quick brown fox jumps"},
            transcribed_user_turns={1: "the quick green fox"},
        )
        result = await metric.compute(ctx)

        assert result.sub_metrics is not None
        assert set(result.sub_metrics.keys()) == {"substitution_rate", "deletion_rate", "insertion_rate"}

        sub_count = result.sub_metrics["substitution_rate"]
        del_count = result.sub_metrics["deletion_rate"]
        ins_count = result.sub_metrics["insertion_rate"]
        ref_words = result.details["reference_words"]
        assert ref_words == 5
        assert sub_count.name == "stt_wer.substitution_rate"
        assert sub_count.details["reference_words"] == 5
        assert sub_count.details["count"] == result.details["total_substitutions"]
        assert del_count.details["count"] == result.details["total_deletions"]
        assert ins_count.details["count"] == result.details["total_insertions"]
        assert sub_count.score == pytest.approx(sub_count.details["count"] / ref_words)
        assert sub_count.normalized_score == sub_count.score

    @pytest.mark.asyncio
    async def test_no_common_turns(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello"},
            transcribed_user_turns={2: "hello"},
        )
        result = await metric.compute(ctx)
        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_bracket_annotations_stripped(self):
        """Bracket annotations like [slow] should be removed before comparison."""
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "[slow] hello world [likely cut off]"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        # After stripping brackets, both should be "hello world"
        assert result.details["wer"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello", 2: "goodbye", 3: "thank you"},
            transcribed_user_turns={1: "hello", 2: "goodbye", 3: "thank you"},
        )
        result = await metric.compute(ctx)
        assert result.details["num_turns"] == 3
        assert result.details["wer"] == 0.0
        assert len(result.details["per_turn_wer"]) == 3

    @pytest.mark.asyncio
    async def test_empty_turns_skipped(self):
        """Turns where reference or hypothesis is empty after stripping are skipped."""
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "[likely interruption]", 2: "hello world"},
            transcribed_user_turns={1: "", 2: "hello world"},
        )
        result = await metric.compute(ctx)
        # Turn 1 should be skipped (empty after stripping)
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_language_config(self):
        metric = STTWERMetric(config={"language": "en"})
        assert metric.language == "en"

    @pytest.mark.asyncio
    async def test_per_turn_errors_included(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "the cat sat on the mat"},
            transcribed_user_turns={1: "the cat sit on a mat"},
        )
        result = await metric.compute(ctx)
        assert 1 in result.details["per_turn_wer"]
        assert 1 in result.details["per_turn_errors"]
        assert "error_summary" in result.details


class TestSTTWERCodeSwitch:
    """Hindi turns with English words: the reconciled WER is the main score."""

    @pytest.mark.asyncio
    async def test_reconciled_score_details_and_sub_metrics(self):
        metric = STTWERMetric(config={"language": "hi"})
        ctx = make_metric_context(
            intended_user_turns={1: "मेरी फ़्लाइट seven बजे है", 2: "मेरी flight कल है"},
            transcribed_user_turns={1: "मेरी फ़्लाइट 7 बजे है", 2: "मेरी fight कल है"},
        )
        result = await metric.compute(ctx)
        assert result.details["strict_wer"] == pytest.approx(round(2 / 9, 3))
        assert result.score == result.details["wer"] == pytest.approx(round(1 / 9, 3))
        assert result.details["per_turn_wer"] == {1: 0.0, 2: 0.25}
        assert result.details["per_turn_code_switch_rescues"] == {
            1: [{"expected": "seven", "actual": "7", "matcher": "secondary_view"}]
        }
        assert result.details["total_substitutions"] == 1

        code_switch = result.details["code_switch"]
        assert code_switch["languages"] == ["hi", "en"]
        # seven, flight. फ़्लाइट is not in the loanword lexicon and matched exactly, so it stays tagged hi.
        assert code_switch["code_switched_words"] == 2
        assert code_switch["rescued_words"] == 1
        assert code_switch["rescues_by_matcher"] == {"secondary_view": 1}
        # Per-language attribution: fight -> flight is an English error.
        assert code_switch["per_language"]["en"] == {"wer": 0.5, "ref_words": 2, "errors": 1}
        assert code_switch["per_language"]["hi"] == {"wer": 0.0, "ref_words": 7, "errors": 0}

        subs = result.sub_metrics
        assert {"wer_hi", "wer_en", "code_switch_word_rate", "code_switch_rescue_rate"} <= subs.keys()
        assert subs["wer_en"].name == "stt_wer.wer_en"
        assert subs["wer_en"].score == pytest.approx(0.5)
        assert subs["code_switch_word_rate"].score == pytest.approx(0.222)
        assert subs["code_switch_rescue_rate"].score == pytest.approx(0.111)

    @pytest.mark.asyncio
    async def test_no_english_words_omits_wer_en(self):
        metric = STTWERMetric(config={"language": "hi"})
        ctx = make_metric_context(
            intended_user_turns={1: "मैं कल आऊंगा"},
            transcribed_user_turns={1: "मैं कल आऊंगा"},
        )
        result = await metric.compute(ctx)
        assert result.details["code_switch"]["per_language"]["en"]["wer"] is None
        assert "wer_en" not in result.sub_metrics
        assert result.details["per_turn_code_switch_rescues"] == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["en", "fr", "de", "es", "it", "nl", "ja"])
    async def test_languages_without_code_switching_are_unchanged(self, language):
        metric = STTWERMetric(config={"language": language})
        ctx = make_metric_context(
            intended_user_turns={1: "I'm at seven thirty", 2: "the quick brown fox"},
            transcribed_user_turns={1: "I am at 7:30", 2: "the quick green fox"},
        )
        result = await metric.compute(ctx)
        assert metric.code_switch_languages == ()
        for key in ("strict_wer", "code_switch", "per_turn_code_switch_rescues"):
            assert key not in result.details
        assert set(result.sub_metrics) == {"substitution_rate", "deletion_rate", "insertion_rate"}

    @pytest.mark.asyncio
    async def test_phonetic_rescue_counts_in_hindi_score(self):
        metric = STTWERMetric(config={"language": "hi"})
        ctx = make_metric_context(
            intended_user_turns={1: "मेरे लैपटॉप में नेटवर्क प्रॉब्लम है"},
            transcribed_user_turns={1: "मेरे लैपटॉप में network problem है"},
        )
        result = await metric.compute(ctx)
        assert metric.code_switch_phonetic
        assert result.details["strict_wer"] == pytest.approx(0.333)
        assert result.details["wer"] == 0.0
        assert result.details["code_switch"]["rescues_by_matcher"] == {"phonetic": 2}
        # नेटवर्क, प्रॉब्लम (rescued). लैपटॉप matched exactly and is not in the lexicon, so it stays hi.
        assert result.details["code_switch"]["per_language"]["en"]["ref_words"] == 2
