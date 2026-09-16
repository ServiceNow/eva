import pytest
import yaml

from eva.user_simulator.cascade import phrases as phrases_module
from eva.user_simulator.cascade.phrases import (
    PHRASES_PATH,
    CallerPhrases,
    candidate_languages,
    load_phrases,
)


@pytest.fixture(autouse=True)
def _clear_phrase_cache():
    phrases_module._cache.clear()
    yield
    phrases_module._cache.clear()


@pytest.fixture
def phrase_file(tmp_path, monkeypatch):
    """Point the loader at a temporary phrase file."""

    def write(data):
        path = tmp_path / "caller_phrases.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        monkeypatch.setattr(phrases_module, "PHRASES_PATH", path)
        return path

    return write


def test_the_shipped_file_has_english():
    # English is the fallback for every language, so it is the one entry that must exist.
    data = yaml.safe_load(PHRASES_PATH.read_text(encoding="utf-8"))

    assert data["en"]["backchannels"]
    assert data["en"]["barge_in_openers"]


def test_a_language_with_its_own_entry_uses_it(phrase_file):
    phrase_file(
        {
            "en": {"backchannels": ["uh-huh"], "barge_in_openers": ["Wait—"]},
            "fr": {"backchannels": ["hmm", "ouais"], "barge_in_openers": ["Attendez—"]},
        }
    )

    assert load_phrases("fr") == CallerPhrases(backchannels=["hmm", "ouais"], barge_in_openers=["Attendez—"])


def test_a_regional_variant_falls_back_to_its_base_language(phrase_file):
    # fr-CA shares its continuers with fr; falling all the way to English would be worse.
    phrase_file(
        {
            "en": {"backchannels": ["uh-huh"], "barge_in_openers": ["Wait—"]},
            "fr": {"backchannels": ["hmm"], "barge_in_openers": ["Attendez—"]},
        }
    )

    assert load_phrases("fr-CA").backchannels == ["hmm"]


def test_an_unknown_language_falls_back_to_english_rather_than_failing(phrase_file):
    # A missing translation should degrade the behavior's realism, not abort the run.
    phrase_file({"en": {"backchannels": ["uh-huh"], "barge_in_openers": ["Wait—"]}})

    assert load_phrases("ja").backchannels == ["uh-huh"]


def test_a_missing_file_yields_an_empty_vocabulary(phrase_file, tmp_path, monkeypatch):
    monkeypatch.setattr(phrases_module, "PHRASES_PATH", tmp_path / "absent.yaml")

    result = load_phrases("en")

    assert result.vocabulary == []


def test_candidate_order_is_most_specific_first():
    assert candidate_languages("fr-CA") == ["fr-CA", "fr", "en"]
    assert candidate_languages("fr") == ["fr", "en"]
    assert candidate_languages("en") == ["en"]


def test_vocabulary_is_everything_that_needs_rendering(phrase_file):
    phrase_file({"en": {"backchannels": ["uh-huh", "mm-hmm"], "barge_in_openers": ["Wait—"]}})

    assert load_phrases("en").vocabulary == ["uh-huh", "mm-hmm", "Wait—"]
