"""Tests for WER text normalization utilities."""

import pytest
from pipecat.frames.frames import Language

from eva.utils.wer_normalization.cjk import (
    ChineseTextNormalizer,
    JapaneseTextNormalizer,
    KoreanTextNormalizer,
)
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
        "digits,words,language",
        [
            ("3", "three", Language.EN),
            ("12", "twelve", Language.EN),
            ("21", "twenty one", Language.EN),
            ("21", "twenty-one", Language.EN),
            ("22", "twenty two", Language.EN),
            ("42", "forty-two", Language.EN),
            ("100", "one hundred", Language.EN),
            ("1000", "one thousand", Language.EN),
            ("2024", "two thousand twenty four", Language.EN),
            ("2024", "twenty twenty four", Language.EN),
            ("EMP343467", "E M P three four three four six seven", Language.EN),
            ("1994-02-11", "nineteen ninety four, zero two, eleven", Language.EN),
            # Hyphen-separated digit IDs / phone numbers (very common when STT writes
            # "919-696-3901" while the user simulator says individual digits).
            ("919-696-3901", "nine one nine six nine six three nine zero one", Language.EN),
            ("899-787", "eight nine nine seven eight seven", Language.EN),
            ("OVH-89B", "O V H eight nine B", Language.EN),
            # User simulator pronounces literal "dash" when spelling IDs;
            # STT writes a hyphen.
            ("WZH-89B", "W Z H dash eight nine B", Language.EN),
            ("1", "un", Language.FR),
            ("1", "un", Language.FR_CA),
            ("1", "une", Language.FR),
            ("1", "une", Language.FR_CA),
            ("3", "trois", Language.FR),
            ("3", "trois", Language.FR_CA),
            ("12", "douze", Language.FR),
            ("12", "douze", Language.FR_CA),
            ("21", "vingt et un", Language.FR),
            ("21", "vingt et un", Language.FR_CA),
            ("30", "trente", Language.FR),
            ("30", "trente", Language.FR_CA),
            ("80", "quatre-vingts", Language.FR),
            ("80", "quatre-vingts", Language.FR_CA),
            ("80", "quatre-vingt", Language.FR),
            ("80", "quatre-vingt", Language.FR_CA),
            ("81", "quatre-vingt-un", Language.FR),
            ("81", "quatre-vingt-un", Language.FR_CA),
            ("90", "quatre-vingt-dix", Language.FR),
            ("90", "quatre-vingt-dix", Language.FR_CA),
            ("92", "quatre-vingt-douze", Language.FR),
            ("92", "quatre-vingt-douze", Language.FR_CA),
            ("99", "quatre-vingt-dix-neuf", Language.FR),
            ("99", "quatre-vingt-dix-neuf", Language.FR_CA),
            ("1er", "premier", Language.FR),
            ("1er", "premier", Language.FR_CA),
            ("1ère", "première", Language.FR),
            ("1ère", "première", Language.FR_CA),
            ("3e", "troisième", Language.FR),
            ("3e", "troisième", Language.FR_CA),
            ("12e", "douzième", Language.FR),
            ("12e", "douzième", Language.FR_CA),
            ("21e", "vingt-et-unième", Language.FR),
            ("21e", "vingt-et-unième", Language.FR_CA),
            ("30e", "trentième", Language.FR),
            ("30e", "trentième", Language.FR_CA),
            ("80e", "quatre-vingtième", Language.FR),
            ("80e", "quatre-vingtième", Language.FR_CA),
            ("81e", "quatre-vingt-unième", Language.FR),
            ("81e", "quatre-vingt-unième", Language.FR_CA),
            ("90e", "quatre-vingt-dixième", Language.FR),
            ("90e", "quatre-vingt-dixième", Language.FR_CA),
            ("92e", "quatre-vingt-douzième", Language.FR),
            ("92e", "quatre-vingt-douzième", Language.FR_CA),
            ("99e", "quatre-vingt-dix-neuvième", Language.FR),
            ("99e", "quatre-vingt-dix-neuvième", Language.FR_CA),
            ("EMP343467", "E M P trois quatre trois quatre six sept", Language.FR),
            ("100", "cent", Language.FR),
            ("Au bâtiment Headquarters, à l'étage FL2.", "au bâtiment headquarters à l'étage fl deux", Language.FR),
        ],
    )
    def test_digits_match_spelled_out(self, digits: str, words: str, language: Language):
        digits_normalized = normalize_text(digits, language=language.value)
        words_normalized = normalize_text(words, language=language.value)
        assert digits_normalized == words_normalized

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
        assert normalize_text(text, Language.EN.value) == expected

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


class TestFrenchTextNormalization:
    def test_filler_words_removed(self):
        result = normalize_text("euh je voudrais un café", Language.FR.value)
        assert "euh" not in result
        # diacritics are stripped by the normalizer, so é → e
        assert "cafe" in result

    def test_abbreviations_expanded(self):
        assert "madame" in normalize_text("mme dupont", Language.FR.value)

    def test_thousand_separator_dot(self):
        assert normalize_text("1.000", Language.FR.value) == normalize_text("1000", Language.FR.value)

    def test_thousand_separator_space(self):
        assert normalize_text("1 000 euros", Language.FR.value) == normalize_text("1000 euros", Language.FR.value)

    def test_decimal_comma_to_dot(self):
        assert normalize_text("1,5", Language.FR.value) == normalize_text("1.5", Language.FR.value)

    def test_markup_removed(self):
        result = normalize_text("bonjour [laughs] comment ca va", Language.FR.value)
        assert "[laughs]" not in result
        assert "bonjour" in result

    def test_diacritics_after_normalisation(self):
        result = normalize_text("Au bâtiment Headquarters, à l'étage FL2.", Language.FR.value)
        assert "batiment" in result or "bâtiment" in result
        assert "headquarters" in result


# ===========================================================================
# Japanese
# ===========================================================================


@pytest.fixture(scope="module")
def ja():
    return JapaneseTextNormalizer()


class TestJapaneseNumbers:
    @pytest.mark.parametrize(
        "kanji,arabic",
        [
            ("十", "10"),
            ("二十", "20"),
            ("三百二十一", "321"),
            ("千二百三十四", "1234"),
            ("一万二千三百四十五", "12345"),
            ("万", "10000"),
            ("一億", "100000000"),
            # Pure-digit concatenation (no units → ID/phone number mode)
            ("一二三", "123"),
            ("〇一二三", "0123"),
        ],
    )
    def test_kanji_to_arabic(self, ja, kanji: str, arabic: str):
        assert ja(kanji) == arabic

    def test_full_width_digits(self, ja):
        assert ja("１２３") == "123"

    def test_full_width_ascii(self, ja):
        assert ja("ＡＢＣＤ") == "abcd"

    def test_filler_removal(self, ja):
        result = ja("えーとよろしくお願いします")
        assert "えーと" not in result
        assert "よろしく" in result

    def test_japanese_punctuation_stripped(self, ja):
        result = ja("こんにちは。今日はいい天気ですね。")
        assert "。" not in result

    def test_markup_removed(self, ja):
        result = ja("お客様[laughs]は三名です")
        assert "[laughs]" not in result
        assert "3" in result


# ===========================================================================
# Chinese
# ===========================================================================


@pytest.fixture(scope="module")
def zh():
    return ChineseTextNormalizer()


class TestChineseNumbers:
    @pytest.mark.parametrize(
        "hanzi,arabic",
        [
            ("三百二十一", "321"),
            ("一千", "1000"),
            ("一亿", "100000000"),
            ("两千", "2000"),
            ("一亿两千万", "120000000"),
            # Traditional Chinese variants
            ("兩千", "2000"),
            ("一億兩千萬", "120000000"),
            ("二十萬", "200000"),
        ],
    )
    def test_hanzi_to_arabic(self, zh, hanzi: str, arabic: str):
        assert zh(hanzi) == arabic

    def test_simplified_and_traditional_equivalent(self, zh):
        assert zh("一亿两千万") == zh("一億兩千萬")

    def test_full_width_ascii(self, zh):
        assert zh("ＡＢＣＤ") == "abcd"

    def test_filler_removal(self, zh):
        result = zh("那个就是对的")
        assert "那个" not in result
        assert "就是" not in result

    def test_markup_removed(self, zh):
        result = zh("价格[music]是三百元")
        assert "[music]" not in result
        assert "300" in result


# ===========================================================================
# Korean
# ===========================================================================


@pytest.fixture(scope="module")
def ko():
    return KoreanTextNormalizer()


class TestKoreanSinoNumbers:
    @pytest.mark.parametrize(
        "sino,arabic",
        [
            ("십", "10"),
            ("삼백이십일", "321"),
            ("이천이십사", "2024"),
            ("만이천삼백사십오", "12345"),
            ("일억", "100000000"),
        ],
    )
    def test_sino_korean_to_arabic(self, ko, sino: str, arabic: str):
        assert ko(sino) == arabic

    def test_ambiguous_syllables_not_converted(self, ko):
        # 이 = "this/two", 일 = "work/one" — must NOT be converted without a unit
        assert ko("이 책") == "이 책"
        assert ko("일하다") == "일하다"


class TestKoreanNativeNumbers:
    @pytest.mark.parametrize(
        "native,arabic",
        [
            ("아홉", "9"),
            ("열", "10"),
            ("열다섯", "15"),
            ("스물하나", "21"),
            ("서른", "30"),
            ("아흔아홉", "99"),
            ("열 하나", "11"),
        ],
    )
    def test_native_korean_to_arabic(self, ko, native: str, arabic: str):
        assert ko(native) == arabic

    def test_filler_removal(self, ko):
        result = ko("음 주문하겠습니다")
        assert result.startswith("주문") or "주문" in result

    def test_markup_removed(self, ko):
        result = ko("고객[laughs]은 스물하나입니다")
        assert "[laughs]" not in result
        assert "21" in result


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
