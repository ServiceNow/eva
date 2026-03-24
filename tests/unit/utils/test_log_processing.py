"""Tests for text normalization functions in voice_agents module."""

from eva.utils.log_processing import normalize_for_comparison, truncate_to_spoken


class TestNormalizeForComparison:
    def test_removes_unicode_special_characters(self):
        """Test that Unicode special characters are removed for robust comparison."""
        # Real example from production: narrow no-break space (\u202f) and non-breaking hyphen (\u2011)
        text_with_unicode = (
            "We have a few choices that leave after two\u202fp.m. Eastern.  \n"
            "The flight at 2:50 p.m. will take you back to Seattle for $330 in Main Cabin.  \n"
            "Your original return cost $310, so the fare difference is twenty dollars.  \n"
            "Because this is a voluntary change, there is a seventy-five\u2011dollar change fee.  \n\n"
            "That adds up to ninety\u2011five dollars total\u2014well under your $100 budget.  \n"
            "Would you like me to book that change for you?"
        )

        text_without_unicode = (
            "We have a few choices that leave after twop.m. Eastern.  \n"
            "The flight at 2:50 p.m. will take you back to Seattle for $330 in Main Cabin.  \n"
            "Your original return cost $310, so the fare difference is twenty dollars.  \n"
            "Because this is a voluntary change, there is a seventy-fivedollar change fee.  \n\n"
            "That adds up to ninetyfive dollars totalwell under your $100 budget.  \n"
            "Would you like me to book that change for you?"
        )

        # Both should normalize to the same alphanumeric string
        normalized_with = normalize_for_comparison(text_with_unicode)
        normalized_without = normalize_for_comparison(text_without_unicode)

        assert normalized_with == normalized_without
        # Verify it's actually normalized (no spaces, punctuation, etc.)
        assert " " not in normalized_with
        assert "." not in normalized_with
        assert "$" not in normalized_with
        assert "\n" not in normalized_with

    def test_removes_all_punctuation_and_whitespace(self):
        """Test that all punctuation and whitespace is removed."""
        text = "Hello, World! How are you? I'm fine."
        result = normalize_for_comparison(text)

        assert result == "helloworldhowareyouimfine"
        assert result.isalnum()

    def test_case_insensitive(self):
        """Test that comparison is case-insensitive."""
        assert normalize_for_comparison("Hello World") == normalize_for_comparison("hello world")
        assert normalize_for_comparison("ABC") == normalize_for_comparison("abc")

    def test_unicode_quotes_normalized(self):
        """Test that various Unicode quote styles are removed."""
        text1 = "\"Hello\" and 'world'"  # ASCII quotes
        text2 = "\u201cHello\u201d and \u2018world\u2019"  # Unicode curly quotes

        assert normalize_for_comparison(text1) == normalize_for_comparison(text2)

    def test_unicode_dashes_normalized(self):
        """Test that various dash types are removed."""
        text1 = "twenty-five"  # ASCII hyphen
        text2 = "twenty\u2011five"  # Non-breaking hyphen
        text3 = "twenty\u2014five"  # Em dash

        result1 = normalize_for_comparison(text1)
        result2 = normalize_for_comparison(text2)
        result3 = normalize_for_comparison(text3)

        assert result1 == result2 == result3 == "twentyfive"

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        assert normalize_for_comparison("") == ""

    def test_only_punctuation(self):
        """Test that string with only punctuation returns empty string."""
        assert normalize_for_comparison("!@#$%^&*()") == ""
        assert normalize_for_comparison("   \n\t  ") == ""

    def test_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        assert normalize_for_comparison("abc123") == "abc123"
        assert normalize_for_comparison("Test123") == "test123"


class TestTruncateToSpoken:
    def test_full_match_with_unicode_differences(self):
        """Test that full match works even with Unicode character differences."""
        audit_text = "The cost is seventy\u2011five dollars."
        pipecat_text = "The cost is seventy-five dollars."

        result = truncate_to_spoken(audit_text, [pipecat_text])

        # Should return the original audit text since it fully matches
        assert result == audit_text

    def test_partial_match_with_unicode(self):
        """Test that partial match works with Unicode differences."""
        audit_text = "The cost is seventy\u2011five dollars and ninety\u2011five cents."
        pipecat_text = "The cost is seventy-five dollars"

        result = truncate_to_spoken(audit_text, [pipecat_text])

        # Should truncate to the matching portion
        assert result is not None
        # The result should contain "seventy\u2011five dollars" but not "ninety\u2011five cents"
        assert "seventy\u2011five" in result
        assert "ninety\u2011five" not in result

    def test_no_match_returns_none(self):
        """Test that no match returns None."""
        audit_text = "Completely different text"
        pipecat_text = "Nothing in common"

        result = truncate_to_spoken(audit_text, [pipecat_text])

        assert result is None
