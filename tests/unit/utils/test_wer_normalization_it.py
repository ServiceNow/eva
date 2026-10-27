"""Italian WER normalization: spoken symbols in spelled IDs and dates."""

import pytest

from eva.utils.wer_normalization.wer_utils import normalize_text


@pytest.mark.parametrize(
    "spoken,written",
    [("A B C trattino uno due tre", "ABC-123"), ("1995 trattino 04 trattino 18", "1995-04-18")],
)
def test_spoken_symbol_matches_written_symbol(spoken, written):
    assert normalize_text(spoken, "it") == normalize_text(written, "it")
