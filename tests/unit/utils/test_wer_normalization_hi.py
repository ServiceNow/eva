"""Hindi WER normalization: nukta and chandrabindu spellings, compounds, Latin-script loanwords."""

import json

import pytest

from eva.utils.wer_normalization.engine import CONFIGS_DIR, LanguageConfig
from eva.utils.wer_normalization.wer_utils import normalize_text


def same(a: str, b: str) -> None:
    na, nb = normalize_text(a, "hi"), normalize_text(b, "hi")
    assert na == nb, f"{a!r} -> {na!r} != {b!r} -> {nb!r}"


@pytest.mark.parametrize(
    "a,b",
    [
        ("ज़रूर", "जरूर"),  # nukta as a combining mark (U+093C)
        ("ज़रूर", "जरूर"),  # precomposed ज़ (U+095B)
        ("फ़्लाइट", "फ्लाइट"),
        ("ऑफ़िस", "ऑफिस"),  # scenario data has both spellings
        ("वेरिफ़िकेशन", "वेरिफिकेशन"),
        ("दस्तावेज़", "दस्तावेज"),
    ],
)
def test_nukta_spelling_matches_plain_spelling(a, b):
    same(a, b)


@pytest.mark.parametrize("a,b", [("हाँ", "हां"), ("पहुँच", "पहुंच"), ("समस्याएँ", "समस्याएं")])
def test_chandrabindu_matches_anusvara(a, b):
    same(a, b)


def test_hyphenated_compound_matches_closed_compound():
    same("ई-मेल", "ईमेल")


def test_reduplication_keeps_the_boundary():
    same("बार-बार", "बार बार")
    same("अभी-अभी", "अभी अभी")


# Short loanwords only: longer ones are left to the phonetic matcher (test_wer_code_switch.py).
@pytest.mark.parametrize(
    "latin,deva",
    [
        ("code", "कोड"),
        ("email", "ईमेल"),
        ("e-mail", "ईमेल"),
        ("office", "ऑफ़िस"),
        ("ID", "आईडी"),
        ("HR", "एचआर"),
    ],
)
def test_latin_loanword_matches_devanagari(latin, deva):
    same(latin, deva)


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("A B C डैश एक दो तीन", "ABC-123"),
        ("A B C अंडरस्कोर एक दो तीन", "ABC_123"),
        ("A B C underscore एक दो तीन", "ABC_123"),
        ("A B C स्लैश एक दो तीन", "ABC/123"),
        ("A B C slash एक दो तीन", "ABC/123"),
    ],
)
def test_spoken_symbol_matches_written_symbol(spoken, written):
    same(spoken, written)


def test_numbers_still_converted():
    same("पाँच", "5")
    same("पांच", "5")
    assert normalize_text("ज़ीरो", "hi") == normalize_text("0", "hi")


def test_plain_devanagari_unchanged():
    assert normalize_text("नमस्ते दोस्त", "hi") == "नमस्ते दोस्त"


def test_loanword_keys_are_lowercase_ascii():
    cfg = LanguageConfig.load("hi")
    with open(CONFIGS_DIR / cfg.spelling_map_path) as f:
        lexicon = json.load(f)
    assert lexicon, "loanword lexicon must not be empty"
    for k in lexicon:
        assert k == k.lower() and k.isascii(), k
