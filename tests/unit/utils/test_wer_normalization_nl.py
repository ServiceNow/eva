"""Dutch WER normalization: hyphenated compounds and spoken symbols."""

import pytest

from eva.utils.wer_normalization.wer_utils import normalize_text


def same(a: str, b: str) -> None:
    na, nb = normalize_text(a, "nl"), normalize_text(b, "nl")
    assert na == nb, f"{a!r} -> {na!r} != {b!r} -> {nb!r}"


@pytest.mark.parametrize("a,b", [("e-mailadres", "emailadres"), ("zorg-verlof", "zorgverlof")])
def test_hyphenated_compound_matches_closed_compound(a, b):
    same(a, b)


def test_acronym_compounds_keep_the_boundary():
    same("HR-Fragen", "HR Fragen")
    same("VPN-Passwort", "VPN Passwort")


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("A B C streepje een twee drie", "ABC-123"),
        ("A B C koppelteken een twee drie", "ABC-123"),
        ("A B C laag streepje X Y Z", "ABC_XYZ"),
    ],
)
def test_spoken_symbol_matches_written_symbol(spoken, written):
    same(spoken, written)
