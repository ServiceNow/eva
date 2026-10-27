"""Canadian French WER normalization: spoken symbols, and the "M" abbreviation in codes."""

import pytest

from eva.utils.wer_normalization.wer_utils import normalize_text


def same(a: str, b: str) -> None:
    na, nb = normalize_text(a, "fr-CA"), normalize_text(b, "fr-CA")
    assert na == nb, f"{a!r} -> {na!r} != {b!r} -> {nb!r}"


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("A B C trait d'union un deux trois", "ABC-123"),
        ("A B C trait d’union un deux trois", "ABC-123"),  # curly apostrophe
        ("P R V tiret S U R G tiret 004", "PRV-SURG-004"),
    ],
)
def test_spoken_symbol_matches_written_symbol(spoken, written):
    same(spoken, written)


def test_letter_m_in_a_code_is_not_expanded_to_monsieur():
    same("M 4 B", "M4B")
    assert "monsieur" not in normalize_text("M 4 B", "fr-CA")


def test_pronoun_me_is_kept():
    assert normalize_text("il me faut", "fr-CA") == "il me faut"


def test_abbreviated_monsieur_still_expands():
    same("M. Dupont", "Monsieur Dupont")
