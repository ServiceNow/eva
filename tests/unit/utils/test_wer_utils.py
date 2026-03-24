"""Tests for WER text normalization utilities."""

from eva.utils.wer_normalization.wer_utils import convert_digits_to_words, normalize_text


class TestConvertDigitsToWords:
    def test_single_digit(self):
        assert convert_digits_to_words("I have 3 cats", "en") == "I have three cats"

    def test_large_number(self):
        result = convert_digits_to_words("There are 42 items", "en")
        assert result == "There are forty-two items"

    def test_multiple_numbers(self):
        result = convert_digits_to_words("Room 12 floor 3", "en")
        assert "twelve" in result
        assert "three" in result

    def test_number_adjacent_to_letters(self):
        """Numbers adjacent to letters should get spaces inserted."""
        result = convert_digits_to_words("A1B", "en")
        assert result == "A one B"

    def test_non_english_returns_unchanged(self):
        assert convert_digits_to_words("3つの猫", "ja") == "3つの猫"
        assert convert_digits_to_words("J'ai 3 chats", "fr") == "J'ai 3 chats"

    def test_no_digits(self):
        assert convert_digits_to_words("hello world", "en") == "hello world"

    def test_zero(self):
        result = convert_digits_to_words("0 items", "en")
        assert result == "zero items"


class TestNormalizeText:
    def test_basic_english(self):
        result = normalize_text("Hello World")
        assert result == "hello world"

    def test_english_with_punctuation(self):
        result = normalize_text("Hello, World!")
        assert result == "hello world"

    def test_non_english_digits_preserved(self):
        """Non-English languages should skip digit-to-word conversion."""
        result = normalize_text("3つの猫", language="ja")
        assert "3" in result or "三" in result  # digit stays or gets normalized by Japanese normalizer
