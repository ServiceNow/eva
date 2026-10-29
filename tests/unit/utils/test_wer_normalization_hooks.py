"""Per-language WER normalization hooks are opt-in.

``char_folds`` and ``join_hyphenated_compounds`` only change the output for
languages whose config enables them. English and every language that does not
opt in must normalize exactly as before. ``spoken_symbols`` is set for every
language, English included.
"""

import pytest

from eva.utils.wer_normalization.engine import LanguageConfig, make_char_folder
from eva.utils.wer_normalization.wer_utils import _normalize_hyphen_groups, _replace_spoken_symbols, normalize_text


def test_empty_char_folds_is_identity():
    fold = make_char_folder({})
    assert fold("Krüger ज़रूर") == "Krüger ज़रूर"


def test_char_folds_keys_must_be_single_characters():
    with pytest.raises(ValueError):
        make_char_folder({"ae": "ä"})


def test_english_config_enables_no_hook():
    cfg = LanguageConfig.load("en")
    assert cfg.char_folds == {}
    assert cfg.join_hyphenated_compounds is False


def test_english_spoken_symbols():
    assert normalize_text("mag underscore mutual", "en") == normalize_text("mag_mutual", "en")
    assert normalize_text("A B C underscore one two three", "en") == normalize_text("ABC_123", "en")
    assert normalize_text("A B C slash one two three", "en") == normalize_text("ABC/123", "en")
    assert normalize_text("slash the price", "en") == "slash the price"
    assert normalize_text("A B C hyphen X Y Z", "en") == "abc hyphen xyz"


@pytest.mark.parametrize("lang", ["en", "fr", "es", "it"])
def test_letter_compounds_kept_unless_enabled(lang):
    assert normalize_text("wishy-washy", lang) != normalize_text("wishywashy", lang)


def test_letter_compound_join_is_off_by_default():
    assert _normalize_hyphen_groups("anmelde-konto") == "anmelde-konto"
    assert _normalize_hyphen_groups("anmelde-konto", join_letter_compounds=True) == "anmeldekonto"


def test_spoken_symbols_only_between_word_tokens():
    table = {"tiret": "-"}
    assert _replace_spoken_symbols("A B C tiret 1 2 3", table, {}) == "A B C-1 2 3"
    assert _replace_spoken_symbols("tiret", table, {}) == "tiret"


def test_code_only_symbols_need_a_code_neighbour():
    assert _replace_spoken_symbols("A B C barra 1 2 3", {}, {"barra": "/"}) == "A B C/1 2 3"
    assert _replace_spoken_symbols("una barra de pan", {}, {"barra": "/"}) == "una barra de pan"
