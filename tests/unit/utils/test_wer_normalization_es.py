"""Spanish WER normalization: spoken symbols, and the "D" abbreviation in codes."""

import pytest

from eva.utils.wer_normalization.wer_utils import normalize_text


def same(a: str, b: str) -> None:
    na, nb = normalize_text(a, "es"), normalize_text(b, "es")
    assert na == nb, f"{a!r} -> {na!r} != {b!r} -> {nb!r}"


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("A B C guion uno dos tres", "ABC-123"),
        ("A B C guión uno dos tres", "ABC-123"),
        ("A B C barra 1 2 3", "A B C/1 2 3"),
    ],
)
def test_spoken_symbol_matches_written_symbol(spoken, written):
    same(spoken, written)


def test_barra_is_kept_in_prose():
    assert "barra" in normalize_text("quiero una barra de pan", "es").split()


def test_letter_d_in_a_code_is_not_expanded_to_don():
    same("D 4", "D4")


def test_verb_da_is_kept():
    assert normalize_text("me da igual", "es") == "me da igual"


def test_abbreviated_don_still_expands():
    same("D. Juan", "Don Juan")
