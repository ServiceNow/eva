"""Japanese CER normalization.

Two properties: the reference (intended TTS text) and the STT transcript
normalize to the same string when they say the same thing, and ordinary
words are not turned into letters or digits.
"""

import jiwer
import pytest

from eva.utils.wer_normalization import normalize_text


def _ja(s: str) -> str:
    return normalize_text(s, "ja")


@pytest.mark.parametrize(
    "spoken,written",
    [
        # katakana letter names vs Latin letters in spelled codes
        ("イー・エム・ピー", "EMP"),
        ("イー、ピー、エックス、ワイ、イー、ケイ", "EPXYEK"),
        ("ワイ、ピー、さん、ジー、ブイ、キュー", "YP3GVQ"),
        ("ディーシー", "DC"),
        ("イーエム", "EM"),
        ("イー・エム・ピー、さん", "EMP3"),
        ("社員番号はイー、エム、ピー、ゼロ、ゼロ、さん、です", "社員番号はEMP003です"),
        ("ブイピーエヌに接続できません", "VPNに接続できません"),
        # digit sequences, ゼロ / 零 / れい
        ("三、一、八、五、ゼロ、四", "318504"),
        ("ゼロ", "0"),
        ("零", "0"),
        ("いち、に、さん", "123"),
        ("ゼロ、れい、いち", "001"),
        # English digits in katakana
        ("ワン、ツー、スリー", "123"),
        ("エー、スリー", "A3"),
        ("スリー、ゼロ", "三〇"),
    ],
)
def test_codes(spoken: str, written: str):
    assert _ja(spoken) == _ja(written)


@pytest.mark.parametrize(
    "text",
    [
        "エース",
        "しかし",
        "田中さんにご連絡します",
        "さんです",
        "スリープモード",
        "フォームに入力",
        "ワンピース",
        "セブンイレブン",
        "ティーを飲みます",
        "ビーフ",
        "イーメール",
        "ワイシャツ",
        "アイスティー",
    ],
)
def test_ordinary_words_not_turned_into_letters(text: str):
    out = _ja(text)
    assert not any(c.isascii() and c.isalnum() for c in out), out


@pytest.mark.parametrize(
    "a,b",
    [
        # commas and spaces are not errors in Japanese
        ("はい、そうです", "はいそうです"),
        ("はい,そうです", "はいそうです"),
        ("はい、 そうです。", "はい そうです"),
        ("EMP 3", "EMP3"),
        ("電話番号は、0120です", "電話番号は0120です"),
    ],
)
def test_punctuation_and_spaces_equivalent(a: str, b: str):
    assert _ja(a).replace(" ", "") == _ja(b).replace(" ", "")


@pytest.mark.parametrize(
    "a,b",
    [
        # same word written with kanji or kana, or different okurigana
        ("名字", "苗字"),
        ("振替", "振り替え"),
        ("何とか", "なんとか"),
        ("合っています", "あっています"),
        # product names: Latin vs katakana, and katakana spelling variants
        ("コンフルエンスにログインできません", "Confluenceにログインできません"),
        ("データドッグ", "Datadog"),
        ("ウィンドウズ", "Windows"),
        ("ウインドウズ", "Windows"),
        ("ワイファイ", "Wi-Fi"),
        ("スラック", "Slack"),
        ("マイクロソフト", "Microsoft"),
        ("フィグマ", "Figma"),
        ("セールスフォース", "Salesforce"),
    ],
)
def test_spelling_variants_equivalent(a: str, b: str):
    assert _ja(a) == _ja(b)


def test_product_names_compare_in_katakana():
    assert _ja("Confluence") == "コンフルエンス"


@pytest.mark.parametrize("hypothesis", ["コンフュエンス", "コンフレンス"])
def test_near_miss_katakana_spelling_gets_partial_credit(hypothesis: str):
    # Against the Latin reference this used to be a 100% character error.
    assert jiwer.cer(_ja("Confluence"), _ja(hypothesis)) < 0.35


@pytest.mark.parametrize("text", ["モデル", "デルタ航空", "Dellwood"])
def test_product_name_inside_a_longer_word_unchanged(text: str):
    assert _ja(text) == text.lower()


def test_fillers_only_removed_at_word_start():
    assert _ja("なんとか") == "なんとか"
    assert _ja("ほんとうに") == "ほんとうに"
    assert _ja("えーと、予約を") == "予約を"
    assert _ja("はい、あのー、予約を") == "はい予約を"


def test_existing_kanji_numbers():
    assert _ja("三百二十一") == "321"
    assert _ja("一万二千三百四十五") == "12345"
    assert "3" in _ja("お客様は三名です")
