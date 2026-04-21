"""Unit tests for pass@k and pass^k computation."""

import pytest

from eva.models.results import MetricScore, PassAtKResult
from eva.utils.pass_at_k import (
    compute_pass_at_k,
    compute_pass_at_k_for_scores,
    compute_pass_power_k,
    parse_trial_record_id,
)


class TestComputePassAtK:
    """Tests for the pass@k formula: 1 - C(n-c, k) / C(n, k)."""

    def test_all_pass(self):
        """When all trials pass, pass@k should be 1.0."""
        assert compute_pass_at_k(n=5, c=5, k=3) == 1.0

    def test_none_pass(self):
        """When no trials pass, pass@k should be 0.0."""
        assert compute_pass_at_k(n=5, c=0, k=1) == 0.0
        assert compute_pass_at_k(n=5, c=0, k=3) == 0.0

    def test_known_value(self):
        """Test against a known computed value."""
        # pass@3 with n=10, c=7: 1 - C(3,3)/C(10,3) = 1 - 1/120 ≈ 0.9917
        result = compute_pass_at_k(n=10, c=7, k=3)
        assert result == pytest.approx(1 - 1 / 120, abs=1e-6)

    def test_k_equals_1(self):
        """pass@1 = c/n (simple probability)."""
        # pass@1 with n=10, c=7: 1 - C(3,1)/C(10,1) = 1 - 3/10 = 0.7
        result = compute_pass_at_k(n=10, c=7, k=1)
        assert result == pytest.approx(0.7, abs=1e-6)

    def test_k_equals_n(self):
        """When k=n, pass@k = 1 if c>0 else 0 (at least one must pass in full draw)."""
        # If c >= 1 and k = n, then n - c < k whenever c >= 1
        assert compute_pass_at_k(n=5, c=1, k=5) == 1.0
        assert compute_pass_at_k(n=5, c=0, k=5) == 0.0

    def test_more_pass_than_fail(self):
        """When n - c < k, pass@k = 1.0 (not enough failures to fill k draws)."""
        assert compute_pass_at_k(n=5, c=4, k=2) == 1.0

    def test_k_zero(self):
        """k=0 means drawing nothing, which trivially passes."""
        assert compute_pass_at_k(n=5, c=3, k=0) == 1.0

    def test_k_exceeds_n_raises(self):
        """K > n should raise ValueError."""
        with pytest.raises(ValueError, match="k .* cannot exceed n"):
            compute_pass_at_k(n=3, c=2, k=5)

    def test_c_exceeds_n_raises(self):
        """C > n should raise ValueError."""
        with pytest.raises(ValueError, match="c .* cannot exceed n"):
            compute_pass_at_k(n=3, c=5, k=1)

    def test_negative_values_raise(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pass_at_k(n=-1, c=0, k=1)
        with pytest.raises(ValueError):
            compute_pass_at_k(n=5, c=-1, k=1)
        with pytest.raises(ValueError):
            compute_pass_at_k(n=5, c=0, k=-1)


class TestComputePassPowerK:
    """Tests for the pass^k formula: C(c, k) / C(n, k)."""

    def test_all_pass(self):
        """When all trials pass, pass^k should be 1.0."""
        assert compute_pass_power_k(n=5, c=5, k=3) == 1.0

    def test_none_pass(self):
        """When no trials pass, pass^k should be 0.0."""
        assert compute_pass_power_k(n=5, c=0, k=1) == 0.0

    def test_known_value(self):
        """Test against a known computed value."""
        # pass^3 with n=10, c=7: C(7,3)/C(10,3) = 35/120 ≈ 0.2917
        result = compute_pass_power_k(n=10, c=7, k=3)
        assert result == pytest.approx(35 / 120, abs=1e-6)

    def test_k_equals_1(self):
        """pass^1 = c/n (simple probability)."""
        result = compute_pass_power_k(n=10, c=7, k=1)
        assert result == pytest.approx(0.7, abs=1e-6)

    def test_c_less_than_k(self):
        """When c < k, pass^k = 0 (not enough passing to fill k draws)."""
        assert compute_pass_power_k(n=10, c=2, k=3) == 0.0

    def test_k_zero(self):
        """k=0 means drawing nothing, which trivially succeeds."""
        assert compute_pass_power_k(n=5, c=3, k=0) == 1.0

    def test_k_exceeds_n_raises(self):
        """K > n should raise ValueError."""
        with pytest.raises(ValueError, match="k .* cannot exceed n"):
            compute_pass_power_k(n=3, c=2, k=5)


class TestParseTrialRecordId:
    """Tests for parsing directory names into (base_id, trial_idx)."""

    def test_simple_trial(self):
        assert parse_trial_record_id("1.2.1_trial_0") == ("1.2.1", 0)

    def test_higher_trial_index(self):
        assert parse_trial_record_id("1.2.1_trial_12") == ("1.2.1", 12)

    def test_no_trial(self):
        assert parse_trial_record_id("1.2.1") == ("1.2.1", None)

    def test_underscore_in_base_id(self):
        """Record IDs with underscores should correctly parse the LAST _trial_N."""
        assert parse_trial_record_id("my_record_trial_3") == ("my_record", 3)

    def test_multiple_trial_in_name(self):
        """Only the last _trial_N should be parsed."""
        assert parse_trial_record_id("record_with_trial_in_name_trial_5") == (
            "record_with_trial_in_name",
            5,
        )

    def test_dots_in_id(self):
        assert parse_trial_record_id("2.3.4_trial_0") == ("2.3.4", 0)

    def test_no_number_after_trial(self):
        """_trial_ without a number should not match."""
        assert parse_trial_record_id("1.2.1_trial_") == ("1.2.1_trial_", None)

    def test_trial_not_at_end(self):
        """_trial_N not at the end should not match."""
        assert parse_trial_record_id("1.2.1_trial_0_extra") == (
            "1.2.1_trial_0_extra",
            None,
        )

    def test_empty_string(self):
        assert parse_trial_record_id("") == ("", None)

    def test_attempt_suffix_stripped(self):
        """_attempt_N suffix should be stripped to get base record ID."""
        assert parse_trial_record_id("1.2.1_attempt_0") == ("1.2.1", None)
        assert parse_trial_record_id("my_record_attempt_3") == ("my_record", None)

    def test_trial_and_attempt_suffix(self):
        """Both _trial_N and _attempt_M suffixes should be handled."""
        assert parse_trial_record_id("1.2.1_trial_0_attempt_1") == ("1.2.1", 0)
        assert parse_trial_record_id("my_record_trial_2_attempt_3") == ("my_record", 2)

    def test_nested_trial(self):
        """Nested path format: record_id/trial_N."""
        assert parse_trial_record_id("1.2.1/trial_0") == ("1.2.1", 0)
        assert parse_trial_record_id("1.2.1/trial_5") == ("1.2.1", 5)

    def test_nested_trial_with_attempt(self):
        """Nested path format with attempt suffix."""
        assert parse_trial_record_id("1.2.1/trial_0_attempt_1") == ("1.2.1", 0)
        assert parse_trial_record_id("my_record/trial_2_attempt_3") == ("my_record", 2)

    def test_nested_trial_dots_in_id(self):
        """Nested path with dots in record ID."""
        assert parse_trial_record_id("2.3.4/trial_0") == ("2.3.4", 0)


class TestComputePassAtKForScores:
    """Tests for the high-level scores → PassAtKResult function."""

    def _make_score(self, normalized: float, error: str | None = None) -> MetricScore:
        return MetricScore(
            name="test_metric",
            score=normalized,
            normalized_score=normalized,
            error=error,
        )

    def test_all_pass(self):
        scores = [self._make_score(0.8), self._make_score(0.9), self._make_score(0.7)]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=2)

        assert isinstance(result, PassAtKResult)
        assert result.n == 3
        assert result.k == 2
        assert result.c == 3
        assert result.pass_at_k == 1.0
        assert result.pass_power_k == 1.0
        assert result.per_trial_passed == [True, True, True]

    def test_none_pass(self):
        scores = [self._make_score(0.2), self._make_score(0.3), self._make_score(0.1)]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=1)

        assert result.c == 0
        assert result.pass_at_k == 0.0
        assert result.pass_power_k == 0.0
        assert result.per_trial_passed == [False, False, False]

    def test_mixed_pass_fail(self):
        scores = [
            self._make_score(0.8),  # pass
            self._make_score(0.3),  # fail
            self._make_score(0.6),  # pass
            self._make_score(0.4),  # fail
            self._make_score(0.9),  # pass
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=2)

        assert result.n == 5
        assert result.c == 3
        assert result.per_trial_passed == [True, False, True, False, True]
        # pass@2 with n=5, c=3: 1 - C(2,2)/C(5,2) = 1 - 1/10 = 0.9
        assert result.pass_at_k == pytest.approx(0.9, abs=1e-6)

    def test_error_trials_excluded(self):
        """Errored trials are filtered out; remaining valid trials are used."""
        scores = [
            self._make_score(0.8),
            self._make_score(0.9, error="LLM timeout"),
            self._make_score(0.7),
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=1)

        assert result is not None
        assert result.n == 2  # Only 2 valid trials (errored one excluded)
        assert result.c == 2  # Both valid trials pass
        assert result.per_trial_passed == [True, True]
        assert result.per_trial_scores == [0.8, 0.7]

    def test_error_trials_insufficient_for_k_returns_none(self):
        """If errors leave fewer than k valid trials, returns None."""
        scores = [
            self._make_score(0.8),
            self._make_score(0.9, error="LLM timeout"),
            self._make_score(0.7),
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=3)

        assert result is None  # Only 2 valid, need 3

    def test_all_trials_errored_returns_none(self):
        """If all trials have errors, returns None."""
        scores = [
            self._make_score(0.8, error="fail1"),
            self._make_score(0.9, error="fail2"),
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=1)

        assert result is None

    def test_error_trials_excluded_k_equals_valid(self):
        """1 error out of 3 trials with k=2 still computes from 2 valid trials."""
        scores = [
            self._make_score(0.8),
            self._make_score(0.9, error="timeout"),
            self._make_score(0.3),  # below threshold
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=2)

        assert result is not None
        assert result.n == 2
        assert result.k == 2
        assert result.c == 1  # Only 0.8 passes, 0.3 fails
        assert result.per_trial_passed == [True, False]

    def test_threshold_boundary(self):
        """Score exactly at threshold should pass."""
        scores = [self._make_score(0.5)]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=1)
        assert result.c == 1
        assert result.per_trial_passed == [True]

    def test_fewer_trials_than_k_returns_none(self):
        """If fewer trials than k (even without errors), returns None."""
        scores = [self._make_score(0.8), self._make_score(0.9)]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=5)

        assert result is None

    def test_uses_normalized_score_over_raw(self):
        """Should prefer normalized_score when available."""
        score = MetricScore(
            name="test",
            score=2.0,  # Raw score (not 0-1)
            normalized_score=0.8,  # Normalized
        )
        result = compute_pass_at_k_for_scores("test", [score], threshold=0.5, k=1)
        assert result.per_trial_scores == [0.8]
        assert result.per_trial_passed == [True]

    def test_falls_back_to_raw_score(self):
        """Should use raw score when normalized_score is None."""
        score = MetricScore(
            name="test",
            score=0.7,
            normalized_score=None,
        )
        result = compute_pass_at_k_for_scores("test", [score], threshold=0.5, k=1)
        assert result.per_trial_scores == [0.7]
        assert result.per_trial_passed == [True]

    def test_skipped_trials_excluded_but_others_still_counted(self):
        """Skipped trials excluded while others still count.

        Verifies pass@k is still computed from the remaining valid trials.
        """
        scores = [
            self._make_score(0.8),  # pass
            MetricScore(name="test_metric", score=None, normalized_score=None, skipped=True),
            self._make_score(0.3),  # fail
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=1)

        assert result is not None
        assert result.n == 2
        assert result.c == 1
        assert result.per_trial_passed == [True, False]

    def test_all_trials_skipped_returns_none(self):
        """If every trial was skipped (no valid scores), the metric contributes no pass@k."""
        scores = [
            MetricScore(name="test_metric", score=None, normalized_score=None, skipped=True),
            MetricScore(name="test_metric", score=None, normalized_score=None, skipped=True),
        ]
        result = compute_pass_at_k_for_scores("test", scores, threshold=0.5, k=1)
        assert result is None
