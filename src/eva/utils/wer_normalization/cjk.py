"""CJK-family text normalizers for WER/CER.

These languages have number systems that don't fit the generic alphabetic
engine (positional kanji, two parallel Korean number systems, etc.) and
need dedicated normalizer classes.

Each class extends BaseTextNormalizer so it shares the same structural
scaffold (markup removal, ignore patterns, whitespace collapsing) while
overriding the number-conversion and script-specific steps.
"""

import re
import unicodedata

from jaconv import jaconv

from eva.utils.wer_normalization.engine import BaseTextNormalizer

# ---------------------------------------------------------------------------
# Shared positional number parser (Japanese, Chinese, Sino-Korean all use it)
# ---------------------------------------------------------------------------


def _parse_positional_number(
    s: str,
    digit_map: dict[str, int],
    small_mult: dict[str, int],
    large_mult: dict[str, int],
) -> str:
    """Convert a numeral span to its Arabic digit string.

    Two modes determined by whether any unit characters are present:
    - Pure-digit spans: simple character concatenation (e.g. 一二三 → "123",
      used for phone numbers, IDs, serial numbers).
    - Positional spans: left-to-right arithmetic where each unit multiplies
      the preceding coefficient (e.g. 三百二十一 → "321").

    Returns the original string if parsing fails.
    """
    has_units = any(c in small_mult or c in large_mult for c in s)

    if not has_units:
        try:
            return "".join(str(digit_map[c]) for c in s)
        except KeyError:
            return s

    try:
        total = 0
        small = 0  # accumulator within current large-unit group
        coeff = 0  # coefficient waiting for the next unit
        for c in s:
            if c in digit_map:
                coeff = digit_map[c]
            elif c in small_mult:
                small += max(coeff, 1) * small_mult[c]
                coeff = 0
            elif c in large_mult:
                # max(..., 1) gives 万 alone → 10_000, matching 十→10 / 百→100 behaviour
                total += max(small + coeff, 1) * large_mult[c]
                small = 0
                coeff = 0
        total += small + coeff
        return str(total)
    except Exception:
        return s


# ===========================================================================
# Japanese
# ===========================================================================

_JP_DIGIT: dict[str, int] = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_JP_SMALL: dict[str, int] = {"十": 10, "百": 100, "千": 1_000}
_JP_LARGE: dict[str, int] = {"万": 10_000, "億": 100_000_000, "兆": 1_000_000_000_000}

_JP_NUM_RE = re.compile(r"[〇零一二三四五六七八九十百千万億兆]+")
_JP_PUNCT_RE = re.compile(r"[。、・「」『』【】〔〕〈〉《》〜～！？…—―]")
# No \b anchors — word boundaries don't apply to Japanese text
_JP_FILLERS = re.compile(r"えーと|ええと|えー|あのー|あの|うーん|うん|ねー|ねえ|んーと|んー|んと|まあ|そうですね")


class JapaneseTextNormalizer(BaseTextNormalizer):
    """Normalize Japanese text for WER/CER calculation.

    Pipeline:
    1. Remove markup and filler words
    2. jaconv.normalize() — half/full-width normalisation, unicode compatibility
    3. Convert kanji numerals → Arabic digits
    4. Strip Japanese punctuation (。、「」 etc.)
    5. Lower-case any remaining ASCII (loanwords, product codes)
    6. Collapse whitespace
    """

    def __init__(self, ignore_patterns: str = ""):
        self._ignore_patterns = ignore_patterns or _JP_FILLERS.pattern

    def __call__(self, s: str) -> str:
        s = self._remove_markup(s)
        s = self._apply_ignore_patterns(s, self._ignore_patterns)
        s = jaconv.normalize(s)
        s = _JP_NUM_RE.sub(lambda m: _parse_positional_number(m.group(), _JP_DIGIT, _JP_SMALL, _JP_LARGE), s)
        s = _JP_PUNCT_RE.sub(" ", s)
        s = s.lower()
        return self._collapse_whitespace(s)


# ===========================================================================
# Chinese (Mandarin — Simplified + Traditional)
# ===========================================================================

# Extends the shared CJK digit set with Simplified (亿, 两) and Traditional
# (萬, 兩) variants so both written forms normalize to the same Arabic output.
_ZH_DIGIT: dict[str, int] = {
    **_JP_DIGIT,
    "两": 2,
    "兩": 2,  # colloquial "two" used in counting (liǎng)
}
_ZH_SMALL: dict[str, int] = {**_JP_SMALL}
_ZH_LARGE: dict[str, int] = {
    **_JP_LARGE,
    "亿": 100_000_000,  # Simplified Chinese for 億
    "萬": 10_000,  # Traditional Chinese for 万
    "兆": 1_000_000_000_000,
}

_ZH_NUM_CHARS = "".join(set(_ZH_DIGIT) | set(_ZH_SMALL) | set(_ZH_LARGE))
_ZH_NUM_RE = re.compile(rf"[{re.escape(_ZH_NUM_CHARS)}]+")

_ZH_PUNCT_RE = re.compile(r"[。，、；：？！…—～「」『』【】《》〈〉·]")
# No \b — Chinese text has no word-boundary separators
_ZH_FILLERS = re.compile(r"那个|那個|就是|然后|然後|嗯|啊|哦|呢|哈|唉")


class ChineseTextNormalizer(BaseTextNormalizer):
    """Normalize Mandarin Chinese text (Simplified + Traditional) for WER/CER.

    Pipeline:
    1. Remove markup and filler words
    2. NFKC unicode normalization (full-width ASCII/digits → half-width)
    3. Convert kanji/hanzi numerals → Arabic digits
    4. Strip Chinese punctuation
    5. Lower-case any remaining ASCII
    6. Collapse whitespace
    """

    def __init__(self, ignore_patterns: str = ""):
        self._ignore_patterns = ignore_patterns or _ZH_FILLERS.pattern

    def __call__(self, s: str) -> str:
        s = self._remove_markup(s)
        s = self._apply_ignore_patterns(s, self._ignore_patterns)
        s = unicodedata.normalize("NFKC", s)
        s = _ZH_NUM_RE.sub(lambda m: _parse_positional_number(m.group(), _ZH_DIGIT, _ZH_SMALL, _ZH_LARGE), s)
        s = _ZH_PUNCT_RE.sub(" ", s)
        s = s.lower()
        return self._collapse_whitespace(s)


# ===========================================================================
# Korean
# ===========================================================================

# --- Sino-Korean (한자어) — positional, same algorithm as Japanese/Chinese ---
# Each entry is a single Hangul syllable used as a number word.
_KO_SINO_DIGIT: dict[str, int] = {
    "영": 0,
    "공": 0,  # zero (영 formal, 공 colloquial)
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_KO_SINO_SMALL: dict[str, int] = {"십": 10, "백": 100, "천": 1_000}
_KO_SINO_LARGE: dict[str, int] = {"만": 10_000, "억": 100_000_000, "조": 1_000_000_000_000}

# Unit chars are unambiguous number-context markers; only convert spans that
# contain at least one, to avoid falsely converting ordinary words (e.g. 이
# also means "this", 일 also means "work").
_KO_SINO_UNITS = set(_KO_SINO_SMALL) | set(_KO_SINO_LARGE)
_KO_SINO_CHARS = "".join(set(_KO_SINO_DIGIT) | _KO_SINO_UNITS)
_KO_SINO_RE = re.compile(rf"[{_KO_SINO_CHARS}]+")


def _convert_ko_sino(m: re.Match) -> str:
    s = m.group()
    if not any(c in _KO_SINO_UNITS for c in s):
        return s  # no unit chars → likely an ordinary word, leave unchanged
    return _parse_positional_number(s, _KO_SINO_DIGIT, _KO_SINO_SMALL, _KO_SINO_LARGE)


# --- Native Korean (고유어) — direct lookup, used for 1–99 ---
# The native system only covers 1–99; larger numbers always use sino-Korean.
# Tens and ones are multi-character Hangul words, not single syllables.
_KO_NATIVE_TENS: dict[str, int] = {
    "열": 10,
    "스물": 20,
    "서른": 30,
    "마흔": 40,
    "쉰": 50,
    "예순": 60,
    "일흔": 70,
    "여든": 80,
    "아흔": 90,
}
_KO_NATIVE_ONES: dict[str, int] = {
    "하나": 1,
    "둘": 2,
    "셋": 3,
    "넷": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
}

# Sort longest-first within each group so alternation matches greedily.
_tens_pat = "|".join(sorted(_KO_NATIVE_TENS, key=len, reverse=True))
_ones_pat = "|".join(sorted(_KO_NATIVE_ONES, key=len, reverse=True))

# Group layout: (1)tens+(2)ones | (3)tens-only | (4)ones-only
_KO_NATIVE_RE = re.compile(rf"({_tens_pat})\s*({_ones_pat})|({_tens_pat})|({_ones_pat})")


def _convert_ko_native(m: re.Match) -> str:
    if m.group(1):  # tens + ones
        return str(_KO_NATIVE_TENS[m.group(1)] + _KO_NATIVE_ONES[m.group(2)])
    if m.group(3):  # tens only
        return str(_KO_NATIVE_TENS[m.group(3)])
    return str(_KO_NATIVE_ONES[m.group(4)])  # ones only


_KO_PUNCT_RE = re.compile(r"[。，、；：？！…—～「」『』【】《》〈〉·。，]")
# Single-syllable words like 아/네/예/에 are excluded — they appear inside
# ordinary Korean words (e.g. 아홉=9) and would cause false stripping.
_KO_FILLERS = re.compile(r"음+|어+|그러니까|그냥|뭐")


class KoreanTextNormalizer(BaseTextNormalizer):
    """Normalize Korean text for WER calculation.

    Handles both number systems:
    - Sino-Korean (한자어): positional, e.g. 삼백이십일 → 321.
      Only converted when a unit syllable (십/백/천/만/억/조) is present,
      to avoid false matches on ambiguous syllables like 이 ("this") or 일 ("work").
    - Native Korean (고유어): direct lookup for 1–99,
      e.g. 스물하나 → 21, 열다섯 → 15.

    Pipeline:
    1. Remove markup and filler words
    2. Convert sino-Korean numeral spans → Arabic digits
    3. Convert native Korean numeral words → Arabic digits
    4. Strip Korean/CJK punctuation
    5. Lower-case any remaining ASCII
    6. Collapse whitespace
    """

    def __init__(self, ignore_patterns: str = ""):
        self._ignore_patterns = ignore_patterns or _KO_FILLERS.pattern

    def __call__(self, s: str) -> str:
        s = self._remove_markup(s)
        s = self._apply_ignore_patterns(s, self._ignore_patterns)
        # Sino-Korean first (positional, unambiguous when units present)
        s = _KO_SINO_RE.sub(_convert_ko_sino, s)
        # Native Korean (tens/ones word lookup, 1–99)
        s = _KO_NATIVE_RE.sub(_convert_ko_native, s)
        s = _KO_PUNCT_RE.sub(" ", s)
        s = s.lower()
        return self._collapse_whitespace(s)
