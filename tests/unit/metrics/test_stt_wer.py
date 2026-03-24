"""Tests for STT WER metric bracket stripping."""

import pytest

from eva.metrics.diagnostic.stt_wer import _BRACKET_PATTERN


@pytest.mark.parametrize(
    "text, expected",
    [
        # No brackets — unchanged
        ("[slow] Hello how are you", "Hello how are you"),
        # Single annotation removed
        ("[assistant interrupts] Hello how are you", "Hello how are you"),
        # Trailing annotation removed
        ("Hello how are you [likely cut off by assistant]", "Hello how are you"),
        # Multiple annotations removed
        ("[assistant interrupts] Hello [speaker likely cut itself off]", "Hello"),
        # Only annotation — becomes empty
        ("[likely interruption]", ""),
    ],
)
def test_bracket_pattern_strips_annotations(text, expected):
    assert _BRACKET_PATTERN.sub("", text).strip() == expected
