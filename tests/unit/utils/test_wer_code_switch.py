"""Code-switch reconciliation: Hindi with English words, rescued only when the English view proves them equal."""

import json

import jiwer
import pytest

from eva.utils.wer_normalization.code_switch import phonetic_match, reconcile_code_switch, tag_token
from eva.utils.wer_normalization.engine import CONFIGS_DIR, LanguageConfig


def reconcile(ref: str, hyp: str):
    return reconcile_code_switch(ref, hyp, "hi", ("en",))


def reconciled_wer(result) -> float:
    return jiwer.wer(" ".join(result.ref_tokens), " ".join(result.hyp_tokens))


@pytest.mark.parametrize(
    "ref,hyp,rescued",
    [
        ("seven बजे", "7 बजे", ("seven", "7")),
        ("twenty five मार्च", "25 मार्च", ("twenty five", "25")),  # a 2 -> 1 region
        ("I'm not sure", "I am not sure", ("m", "am")),
    ],
)
def test_english_view_rescues_the_region(ref, hyp, rescued):
    result = reconcile(ref, hyp)
    assert result.strict_output.wer > 0
    assert [(r.expected, r.actual) for r in result.rescues] == [rescued]
    assert reconciled_wer(result) == 0.0


@pytest.mark.parametrize(
    "ref,hyp",
    [
        ("मुझे 7 बजे", "मुझे seven बजे"),  # reference digit is not attested as English: the user may have said सात
        ("मुझे सात बजे", "मुझे seven बजे"),  # no Hindi -> English cross-comparison
        ("मेरी flight कल है", "मेरी fight कल है"),  # real STT error
        ("मैं कल आऊँगा", "मैं call आऊँगा"),  # कल means tomorrow
    ],
)
def test_not_rescued(ref, hyp):
    result = reconcile(ref, hyp)
    assert result.rescues == []
    assert reconciled_wer(result) == pytest.approx(result.strict_output.wer)


def test_region_cutting_through_an_english_number_is_rejected():
    # English view of the reference is "25": the error region "twenty" -> "20"
    # covers only half of it, so it can't be projected.
    result = reconcile("twenty five मार्च", "20 five मार्च")
    assert result.rescues == []


def test_region_longer_than_four_tokens_is_not_rescued():
    result = reconcile("one two three four five", "1 2 3 4 5")
    assert result.rescues == []


def test_reconciled_never_exceeds_strict():
    result = reconcile("मेरी flight कल seven बजे है", "मेरी fight कल 7 बजे है")
    assert [(r.expected, r.actual) for r in result.rescues] == [("seven", "7")]
    assert reconciled_wer(result) == pytest.approx(1 / 6)
    assert result.strict_output.wer == pytest.approx(2 / 6)


def test_adjacent_errors_form_one_region_rescued_only_as_a_whole():
    # "flight seven" vs "fight 7" is one region; its English views differ, so nothing is rescued.
    result = reconcile("मेरी flight seven बजे है", "मेरी fight 7 बजे है")
    assert result.rescues == []


@pytest.mark.parametrize(
    "token,tag",
    [("seven", "en"), ("कोड", "en"), ("टाइम", "en"), ("फ्लाइट", "hi"), ("बजे", "hi"), ("25", "num")],
)
def test_tag_token(token, tag):
    assert tag_token(token, "hi", "en") == tag


# Phase 0 / Phase 1 examples, with the phonetic matcher on as in hi.json:
# (reference, STT, strict WER, reconciled WER, English reference words)
@pytest.mark.parametrize(
    "ref,hyp,strict,after,english",
    [
        ("मुझे आईटी टीम से बात करनी है", "मुझे IT टीम से बात करनी है", 0.0, 0.0, 2),
        ("can I book it", "can I book it", 0.0, 0.0, 4),
        ("मेरी फ़्लाइट seven बजे है", "मेरी फ़्लाइट 7 बजे है", 0.2, 0.0, 1),
        ("मुझे twenty five मार्च की booking चाहिए", "मुझे 25 मार्च की बुकिंग चाहिए", 3 / 7, 0.0, 3),
        ("I'm not sure, क्या आप check कर सकते हैं", "I am not sure क्या आप check कर सकते हैं", 0.1, 0.0, 5),
        ("मुझे 7 बजे की फ़्लाइट चाहिए", "मुझे seven बजे की फ़्लाइट चाहिए", 1 / 6, 1 / 6, 0),
        ("मेरी flight कल है", "मेरी fight कल है", 0.25, 0.25, 1),
    ],
)
def test_plan_examples(ref, hyp, strict, after, english):
    result = reconcile_code_switch(ref, hyp, "hi", ("en",), phonetic=True)
    assert result.strict_output.wer == pytest.approx(strict)
    assert reconciled_wer(result) == pytest.approx(after)
    assert result.ref_langs.count("en") == english


# --- Phase 2: phonetic matcher -------------------------------------------------


@pytest.mark.parametrize(
    "ref_word,hyp_word",
    [
        ("कैंसल", "cancel"),
        ("फ्लाइट", "flight"),
        ("नेटवर्क", "network"),
        ("प्रॉब्लम", "problem"),
        ("कन्फर्म", "confirm"),
        ("पासवर्ड", "password"),
        ("अकाउंट", "account"),
        ("लैपटॉप", "laptop"),
        ("वीपीएन", "vpn"),
        # soft g sounds like ज; hard g still matches
        ("मैनेजर", "manager"),
        ("चार्ज", "charge"),
        ("सर्जरी", "surgery"),
        ("टारगेट", "target"),
        # s between vowels can sound like ज़ (folded to ज); plain s still matches
        ("बिजनेस", "business"),
        ("प्लीज", "please"),
        ("बेसिक", "basic"),
        # anusvara before a labial is m
        ("नंबर", "number"),
        ("कैंपस", "campus"),
        ("सितंबर", "september"),
    ],
)
def test_phonetic_match(ref_word, hyp_word):
    assert phonetic_match(ref_word, hyp_word)


@pytest.mark.parametrize("latin,deva", [("booking", "बुकिंग"), ("march", "मार्च"), ("confirmation", "कन्फर्मेशन")])
def test_phonetic_match_works_in_both_directions(latin, deva):
    assert phonetic_match(deva, latin)
    assert phonetic_match(latin, deva)


@pytest.mark.parametrize(
    "ref_word,hyp_word",
    [
        ("कल", "call"),  # stoplist: कल = tomorrow
        ("बस", "bus"),
        ("मैं", "main"),
        ("पर", "par"),
        ("मुझे", "mujhe"),  # fewer than 3 consonants
        ("network", "network"),  # one word in each script
        ("नेटवर्क", "नेटवर्क"),
        ("नेटवर्क", "netflix"),
    ],
)
def test_phonetic_no_match(ref_word, hyp_word):
    assert not phonetic_match(ref_word, hyp_word)


@pytest.mark.parametrize(
    "ref,hyp,strict,after",
    [
        ("मेरे लैपटॉप में नेटवर्क प्रॉब्लम है", "मेरे लैपटॉप में network problem है", 1 / 3, 0.0),
        ("कृपया मेरा अपॉइंटमेंट कन्फर्म कीजिए", "कृपया मेरा appointment confirm कीजिए", 0.4, 0.0),
        (
            "नमस्ते, मुझे अपनी उड़ान की तारीख 25 मार्च करनी है। मेरा कन्फर्मेशन कोड तैयार है।",
            "नमस्ते, मुझे अपनी उड़ान की तारीख 25 march करनी है। मेरा confirmation code तैयार है।",
            2 / 15,
            0.0,
        ),
        ("मैं कल आऊँगा", "मैं call आऊँगा", 1 / 3, 1 / 3),
        ("मैं बस से आऊँगा", "मैं bus से आऊँगा", 0.25, 0.25),
    ],
)
def test_phonetic_plan_examples(ref, hyp, strict, after):
    result = reconcile_code_switch(ref, hyp, "hi", ("en",), phonetic=True)
    assert result.strict_output.wer == pytest.approx(strict)
    assert reconciled_wer(result) == pytest.approx(after)
    assert all(r.matcher == "phonetic" for r in result.rescues)


def test_phonetic_rescue_tags_the_reference_word_english():
    result = reconcile_code_switch("मेरे नेटवर्क में", "मेरे network में", "hi", ("en",), phonetic=True)
    assert result.ref_langs == ["hi", "en", "hi"]
    assert result.hyp_langs == ["hi", "en", "hi"]


def test_phonetic_is_off_by_default():
    assert reconcile("मेरे नेटवर्क में", "मेरे network में").rescues == []


def test_phonetic_on_only_for_hindi():
    assert not LanguageConfig.model_fields["code_switch_phonetic"].default
    enabled = {
        path.name
        for path in CONFIGS_DIR.glob("*.json")
        if isinstance(data := json.loads(path.read_text()), dict) and data.get("code_switch_phonetic")
    }
    assert enabled == {"hi.json"}
