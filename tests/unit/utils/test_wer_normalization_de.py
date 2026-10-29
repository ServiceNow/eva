"""German WER normalization: umlauts, hyphenated compounds, spoken symbols."""

import pytest

from eva.utils.wer_normalization.wer_utils import normalize_text


def same(a: str, b: str) -> None:
    na, nb = normalize_text(a, "de"), normalize_text(b, "de")
    assert na == nb, f"{a!r} -> {na!r} != {b!r} -> {nb!r}"


@pytest.mark.parametrize(
    "a,b",
    [("Krüger", "Krueger"), ("Müller", "Mueller"), ("Größe", "Groesse"), ("Straße", "Strasse"), ("Ärzte", "Aerzte")],
)
def test_umlaut_and_eszett_match_their_digraphs(a, b):
    same(a, b)


@pytest.mark.parametrize(
    "words,digits",
    [("fünf", "5"), ("zwölf", "12"), ("dreißig", "30"), ("fünfundzwanzig", "25"), ("fünfhundert", "500")],
)
def test_number_words_with_umlauts_still_parse(words, digits):
    same(words, digits)


def test_words_with_ue_are_unchanged():
    assert normalize_text("neue Feuer", "de") == "neue feuer"


@pytest.mark.parametrize("a,b", [("Anmelde-Konto", "Anmeldekonto"), ("E-Mail-Adresse", "Emailadresse")])
def test_hyphenated_compound_matches_closed_compound(a, b):
    same(a, b)


def test_acronym_compounds_keep_the_boundary():
    # STT often writes these open ("HR Fragen"), so the acronym stays a separate token.
    same("HR-Fragen", "HR Fragen")
    same("VPN-Passwort", "VPN Passwort")


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("A B C Bindestrich eins zwei drei", "ABC-123"),
        ("P R V Bindestrich S U R G Bindestrich 004", "PRV-SURG-004"),
        ("Ich brauche read Unterstrich only Zugriff", "Ich brauche read_only Zugriff"),
        ("Ich brauche read underscore only Zugriff", "Ich brauche Read only Zugriff"),
        ("A B C Strich X Y Z", "ABC-XYZ"),
    ],
)
def test_spoken_symbol_matches_written_symbol(spoken, written):
    same(spoken, written)


@pytest.mark.parametrize(
    "text,kept", [("das ist ein Strich durch die Rechnung", "strich"), ("vier minus zwei", "minus")]
)
def test_ordinary_words_are_kept_in_prose(text, kept):
    assert kept in normalize_text(text, "de").split()
