"""French WER normalization: spoken symbols, and the "M" abbreviation in codes.

Canadian French (fr-CA) has no config of its own and uses fr.json, so every
test runs for both.
"""

import pytest

from eva.utils.wer_normalization.wer_utils import _make_normalizer, normalize_text

LANGS = ["fr", "fr-CA"]


def same(a: str, b: str, lang: str) -> None:
    na, nb = normalize_text(a, lang), normalize_text(b, lang)
    assert na == nb, f"{a!r} -> {na!r} != {b!r} -> {nb!r} ({lang})"


def test_fr_ca_uses_the_french_config():
    assert _make_normalizer("fr-CA").config.code == "fr"


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(
    "spoken,written",
    [
        ("A B C trait d'union un deux trois", "ABC-123"),
        ("A B C trait d’union un deux trois", "ABC-123"),  # curly apostrophe
        ("P R V tiret S U R G tiret 004", "PRV-SURG-004"),
        ("A B C tiret bas X Y Z", "ABC_XYZ"),
        ("A B C tiret du bas X Y Z", "ABC_XYZ"),
    ],
)
def test_spoken_symbol_matches_written_symbol(spoken, written, lang):
    same(spoken, written, lang)


@pytest.mark.parametrize("lang", LANGS)
def test_barre_is_kept_in_prose(lang):
    assert "barre" in normalize_text("la barre est haute", lang).split()


@pytest.mark.parametrize("lang", LANGS)
def test_letter_m_in_a_code_is_not_expanded_to_monsieur(lang):
    same("M 4 B", "M4B", lang)
    assert "monsieur" not in normalize_text("M 4 B", lang)


@pytest.mark.parametrize("lang", LANGS)
def test_pronoun_me_is_kept(lang):
    assert normalize_text("il me faut", lang) == "il me faut"


@pytest.mark.parametrize("lang", LANGS)
def test_abbreviated_monsieur_still_expands(lang):
    same("M. Dupont", "Monsieur Dupont", lang)


@pytest.mark.parametrize("lang", LANGS)
def test_fillers_removed(lang):
    assert normalize_text("euh mm ah je voudrais réserver", lang) == normalize_text("je voudrais réserver", lang)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(
    "spoken,written",
    [
        # phone numbers read in pairs, with no commas between the pairs
        ("zéro six douze trente-quatre cinquante-six soixante-dix-huit", "0612345678"),
        ("zéro un quatre-vingt-un soixante-quinze quatre-vingt-dix-neuf", "01 81 75 99"),
        ("trente-quatre soixante-dix", "3470"),
        ("douze quatre-vingt-dix", "1290"),
        ("douze quatre-vingts", "1280"),
    ],
)
def test_vigesimal_numbers_after_another_number(spoken, written, lang):
    assert normalize_text(spoken, lang).replace(" ", "") == normalize_text(written, lang).replace(" ", "")


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(
    "spoken,written",
    [
        ("soixante-dix", "70"),
        ("quatre-vingt-dix-sept", "97"),
        ("mille neuf cent quatre-vingt-dix", "1990"),
        ("un deux trois", "123"),
        ("trente-quatre soixante", "3460"),
    ],
)
def test_numbers_unchanged_by_the_vigesimal_fix(spoken, written, lang):
    assert normalize_text(spoken, lang) == normalize_text(written, lang)
