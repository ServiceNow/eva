"""Tests for WER text normalization utilities."""

import pytest

from eva.utils.wer_normalization.wer_utils import normalize_text


class TestNormalizeText:
    def test_basic_english(self):
        assert normalize_text("Hello World") == "hello world"

    def test_english_with_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_non_english_digits_preserved(self):
        result = normalize_text("3つの猫", language="ja")
        # digit stays or is converted by the Japanese normalizer
        assert "3" in result or "三" in result


class TestWordDigitEquivalence:
    """Reference (digit form) and ASR (spelled-out) must normalize to the same string."""

    @pytest.mark.parametrize(
        "digits,words",
        [
            ("3", "three"),
            ("12", "twelve"),
            ("21", "twenty one"),
            ("21", "twenty-one"),
            ("22", "twenty two"),
            ("42", "forty-two"),
            ("100", "one hundred"),
            ("1000", "one thousand"),
            ("2024", "two thousand twenty four"),
            ("2024", "twenty twenty four"),
            ("EMP343467", "E M P three four three four six seven"),
            ("1994-02-11", "nineteen ninety four, zero two, eleven"),
            # Hyphen-separated digit IDs / phone numbers (very common when STT writes
            # "919-696-3901" while the user simulator says individual digits).
            ("919-696-3901", "nine one nine six nine six three nine zero one"),
            ("899-787", "eight nine nine seven eight seven"),
            ("OVH-89B", "O V H eight nine B"),
            # User simulator pronounces literal "dash" when spelling IDs;
            # STT writes a hyphen.
            ("WZH-89B", "W Z H dash eight nine B"),
        ],
    )
    def test_digits_match_spelled_out(self, digits: str, words: str):
        assert normalize_text(digits) == normalize_text(words)

    def test_spelled_dash_drops_in_id_readout(self):
        # User simulator readout with "dash" between groups should not include
        # the literal "dash" token after normalization (it's been dropped).
        assert "dash" not in normalize_text("P R V dash S U R G dash zero zero four")

    def test_pure_letter_hyphen_compounds_unchanged(self):
        # Hyphen-concatenation must not collapse pure-letter compounds.
        assert normalize_text("wishy-washy") == normalize_text("wishy washy")


class TestOrdinalNotConvertedToSaint:
    """Regression: '21st' was being converted to '21 saint'.

    The digit→word step split '21st' into 'twenty-one st', and Whisper's
    '\\bst\\b' -> 'saint' abbreviation expansion then matched the now-standalone
    'st'.
    """

    def test_21st_not_converted_to_saint(self):
        result = normalize_text("21st")
        assert result == "21st"
        assert "saint" not in result

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("the 21st of June", "the 21st of june"),
            ("I will arrive on the 21st", "i will arrive on the 21st"),
            ("22nd", "22nd"),
            ("3rd", "3rd"),
            ("4th", "4th"),
        ],
    )
    def test_ordinals_preserved(self, text: str, expected: str):
        assert normalize_text(text) == expected

    @pytest.mark.parametrize(
        "digits,words",
        [
            ("21st", "twenty-first"),
            ("21st", "twenty first"),
            ("22nd", "twenty-second"),
            ("3rd", "third"),
        ],
    )
    def test_ordinal_digits_match_spelled_out(self, digits: str, words: str):
        assert normalize_text(digits) == normalize_text(words)

    def test_st_abbreviation_still_expands(self):
        # The Whisper "St. -> Saint" expansion must still work for actual usage.
        assert "saint" in normalize_text("St. Mary")


class TestTimeNotMerged:
    """Regression: '10:00 AM' was being normalized to '100 am'.

    The digit→word step produced 'ten:zero AM' -> 'ten zero am' which Whisper's
    process_words then concatenated as '10' + '0' = '100'.
    """

    def test_10_00_am_not_merged_to_100_am(self):
        assert normalize_text("10:00 AM") != "100 am"


class TestKnownLimitations:
    """Cases that do not round-trip yet. Marked xfail so they run and surface if a future change happens to fix them."""

    @pytest.mark.xfail(
        reason=(
            "3+ hyphen groups with mixed letter/digit components: digit form "
            "concatenates everything ('prvsurg 4') while spelled form keeps "
            "letter groups separate ('prv surg 4'). Aggregate WER is still "
            "lower with the hyphen-concat than without."
        )
    )
    def test_three_group_alphanumeric_id_round_trip(self):
        assert normalize_text("PRV-SURG-004") == normalize_text("P R V dash S U R G dash zero zero four")

    @pytest.mark.xfail(
        reason=(
            "Whisper's number normalizer treats 'o' as zero (it's in self.zeros "
            "alongside 'oh'/'zero'), so 'ten o clock a m' -> '100 clock a m'. "
            "Meanwhile the digit form '10:00 AM' -> '10 am' (process_words drops "
            "'00' because Fraction(0) is falsy)."
        )
    )
    def test_oclock_round_trip(self):
        assert normalize_text("Ten o clock A M") == normalize_text("10:00 AM")

    @pytest.mark.xfail(
        reason=(
            "Non-ISO date format ('01-01-2026') is concatenated by the "
            "hyphen-digit-group rule ('1012026'). Only the strict ISO form "
            "YYYY-MM-DD is recognized as a date and split into components."
        )
    )
    def test_us_date_round_trip(self):
        assert normalize_text("01-01-2026") == normalize_text("January first twenty twenty six")
