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

# Many number syllables are also ordinary words or surnames (이 "this",
# 일 "work", 천/백/조 surnames, 만나다, 조금, 구조, 사회, 공원), so a span is
# converted only when the context makes it a number. The rules are below.
_KO_SINO_UNITS = set(_KO_SINO_SMALL) | set(_KO_SINO_LARGE)
_KO_DIGIT_CHARS = "".join(_KO_SINO_DIGIT)
_KO_UNIT_CHARS = "".join(_KO_SINO_UNITS)
_HANGUL_RE = re.compile(r"[가-힣]")

# A Sino-Korean number phrase: Arabic numbers (optionally with thousands
# commas) and Sino syllables. A single space is allowed after a unit syllable
# (만 이천, 1만 2천), as in standard Korean spacing.
_KO_ATOM = rf"(?:\d+(?:,\d{{3}})*|[{_KO_DIGIT_CHARS}{_KO_UNIT_CHARS}])"
_KO_PHRASE_RE = re.compile(rf"{_KO_ATOM}+(?:(?<=[{_KO_UNIT_CHARS}]) {_KO_ATOM}+)*")
_KO_ATOM_RE = re.compile(_KO_ATOM)

# Counters and units that mark the preceding word as a number.
_KO_COUNTERS = (
    "개월",
    "시간",
    "퍼센트",
    "달러",
    "킬로",
    "미터",
    "인분",
    "번째",
    "가지",
    "마리",
    "사람",
    "프로",
    "원",
    "명",
    "개",
    "년",
    "월",
    "일",
    "시",
    "분",
    "초",
    "번",
    "호",
    "층",
    "회",
    "세",
    "살",
    "잔",
    "장",
    "권",
    "대",
    "병",
    "점",
    "배",
    "주",
    "달",
    "박",
    "곳",
    "엔",
)
_KO_COUNTER_RE = re.compile("|".join(sorted(_KO_COUNTERS, key=len, reverse=True)))
# Counters that may follow a single digit syllable directly (삼월, 일년, 이층).
# Most counters are left out because the pair is usually a word: 이번 "this
# time", 사회 "society", 공원 "park", 사원 "employee", 이동 "move".
# 이분 ("this person") and 구분 ("category") are excluded for 분.
_KO_SINGLE_DIGIT_COUNTERS: dict[str, frozenset[str]] = {
    "월": frozenset(),
    "년": frozenset(),
    "층": frozenset(),
    "분": frozenset({"이", "구"}),
    "초": frozenset(),
}
# Unit syllables that are also common surnames; alone they are a number only
# when a counter follows (천 원, 백 명), not in 천민수, 천 씨, 백지수.
_KO_SURNAME_UNITS = frozenset({"천", "백", "만", "억", "조"})
# Two-syllable digit strings ending in 일 that are words, not "N일".
_KO_NOT_DAY = frozenset({"오일", "일일"})
# A month just before the phrase: then a trailing 일 is the day counter
# (삼월 이십일 = 3월 20일, not 21).
_KO_MONTH_BEFORE_RE = re.compile(rf"(?:\d|[{_KO_DIGIT_CHARS}십]|유|시)월\s*$")
# Irregular month names.
_KO_IRREGULAR_MONTH_RE = re.compile(r"(?<![가-힣])(유|시)월")
_KO_IRREGULAR_MONTHS = {"유": "6월", "시": "10월"}


def _ko_atoms(s: str) -> list[int | str]:
    return [int(a.replace(",", "")) if a[0].isdigit() else a for a in _KO_ATOM_RE.findall(s)]


def _ko_parse_positional(atoms: list[int | str]) -> int | None:
    """Parse Sino-Korean / mixed atoms (1만 2천, 이십오) into a value.

    Returns None when the sequence is not a well-formed number (two digits in
    a row, units not in decreasing order, a zero coefficient), so ordinary
    words such as 천천히 and spans with a trailing counter (이십오일) are
    detected.
    """
    total = 0
    group = 0
    coeff: int | None = None
    last_small = 10**4
    last_large = 10**16
    for a in atoms:
        if isinstance(a, int) or a in _KO_SINO_DIGIT:
            if coeff is not None:
                return None
            coeff = a if isinstance(a, int) else _KO_SINO_DIGIT[a]
            if coeff == 0:
                return None
        elif a in _KO_SINO_SMALL:
            unit = _KO_SINO_SMALL[a]
            if unit >= last_small or (coeff is not None and coeff > 9):
                return None
            group += (coeff or 1) * unit
            coeff, last_small = None, unit
        else:
            unit = _KO_SINO_LARGE[a]
            if unit >= last_large:
                return None
            value = group + (coeff or 0)
            if coeff is not None and coeff > 9 and group:
                return None
            if value >= 10_000:
                return None
            total += (value or 1) * unit
            group, coeff, last_small, last_large = 0, None, 10**4, unit
    if coeff is not None:
        if (group and coeff > 9) or (total and coeff >= last_large):
            return None
        group += coeff
    return total + group


def _ko_counter_after(after: str) -> bool:
    return bool(_KO_COUNTER_RE.match(after.lstrip(" ")))


def _ko_convert_phrase(text: str, before: str, after: str) -> str:
    """Convert one number phrase given the original text around it."""
    atoms = _ko_atoms(text)
    sino = [a for a in atoms if isinstance(a, str)]
    if not sino:
        return text  # pure Arabic
    has_unit = any(a in _KO_SINO_UNITS for a in sino)
    has_arabic = len(sino) < len(atoms)
    left_free = not before or not _HANGUL_RE.match(before[-1])
    month_before = bool(_KO_MONTH_BEFORE_RE.search(before))
    compact = text.replace(" ", "")

    if not has_unit and not has_arabic:
        # Digit syllables only: dates, phone numbers, IDs.
        if not left_free:
            return text
        n = len(compact)
        if n >= 2 and compact.endswith("일") and (month_before or (n == 2 and compact not in _KO_NOT_DAY)):
            if compact[0] in "영공":
                return text
            return "".join(str(_KO_SINO_DIGIT[c]) for c in compact[:-1]) + "일"
        if n == 1:
            m = _KO_COUNTER_RE.match(after)
            if m and m.group() in _KO_SINGLE_DIGIT_COUNTERS and compact not in _KO_SINGLE_DIGIT_COUNTERS[m.group()]:
                if compact not in "영공":
                    return str(_KO_SINO_DIGIT[compact])
            return text
        if n >= 4 or (n == 3 and any(c in "영공" for c in compact)):
            return "".join(str(_KO_SINO_DIGIT[c]) for c in compact)
        return text

    value = _ko_parse_positional(atoms)
    if compact.endswith("일") and len(atoms) >= 2 and (value is None or month_before):
        day = _ko_parse_positional(atoms[:-1])
        if day is not None:
            return f"{day}일"
    if value is None:
        return text

    attached = bool(after) and bool(_HANGUL_RE.match(after[0]))
    has_digit = any(isinstance(a, int) or a in _KO_SINO_DIGIT for a in atoms)
    if not has_arabic:
        if attached and not _KO_COUNTER_RE.match(after):
            # 만나다, 조금, 백화점, 이만큼, 오만하다, 구조를
            return text
        if not has_digit and compact in _KO_SURNAME_UNITS and not _ko_counter_after(after):
            # 천 씨, 백 선생님
            return text
        if "조" in sino and not _ko_counter_after(after):
            # 구조 "structure", 이조
            return text
    return str(value)


def _ko_split_groups(m: re.Match) -> str:
    """Convert a phrase, merging space-separated pieces only when valid."""
    text, base = m.group(), m.start()
    pieces: list[tuple[int, int]] = []
    pos = 0
    for piece in text.split(" "):
        pieces.append((pos, pos + len(piece)))
        pos += len(piece) + 1

    # Group pieces: a piece joins the previous group when it contains a unit
    # or is Arabic, and the merged phrase is a valid number (만 이천 → 12000).
    groups: list[tuple[int, int]] = []
    for start, end in pieces:
        piece = text[start:end]
        if groups:
            g_start, _ = groups[-1]
            candidate = text[g_start:end]
            joinable = any(c in _KO_SINO_UNITS for c in piece) or piece.replace(",", "").isdigit()
            if joinable and _ko_parse_positional(_ko_atoms(candidate)) is not None:
                groups[-1] = (g_start, end)
                continue
        groups.append((start, end))

    out = []
    prev = 0
    for start, end in groups:
        out.append(text[prev:start])
        out.append(
            _ko_convert_phrase(text[start:end], m.string[: base + start], m.string[base + end :]),
        )
        prev = end
    out.append(text[prev:])
    return "".join(out)


# Space-separated runs of digit syllables read one by one (일 이 삼,
# 공일공 일이삼사): at least three words, or at least four syllables in all.
# The last word may carry a particle or the copula (일 이 삼입니다).
# Possessive ++ so that a particle is never carved out of a digit word.
_KO_DIGIT_WORDS = rf"(?<![가-힣\d])[{_KO_DIGIT_CHARS}]++(?: [{_KO_DIGIT_CHARS}]++)+"
_KO_DIGIT_SEQ_RE = re.compile(
    rf"{_KO_DIGIT_WORDS}(?=입니|이에|이요|이고|이며|까지|을|를|은|는|이|가|요|에)|{_KO_DIGIT_WORDS}(?![가-힣])"
)


def _convert_ko_digit_seq(m: re.Match) -> str:
    words = m.group().split(" ")
    if len(words) < 3 and sum(map(len, words)) < 4:
        return m.group()
    return "".join(str(_KO_SINO_DIGIT[c]) for c in m.group().replace(" ", ""))


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
# Attributive forms used before a counter (한 개, 두 시, 열두 시, 스무 살).
_KO_NATIVE_ATTR: dict[str, int] = {"한": 1, "두": 2, "세": 3, "네": 4}
_KO_NATIVE_ATTR_TENS: dict[str, int] = {"스무": 20}
# Counters used with native numbers.
_KO_NATIVE_COUNTER_RE = re.compile(
    r" ?(?:시간|시|명|개|살|번|잔|장|마리|권|대|병|곳|가지|사람|달|군데|송이|켤레|그릇|벌|채)"
)
# What may directly follow a native number word: a counter or a particle.
# Prevents 열심히, 열차, 넷플릭스, 하나님, 둘러보다 from being converted.
_KO_NATIVE_FOLLOW_RE = re.compile(
    r"(?:시간|시|명|개|살|번|잔|장|마리|권|대|병|곳|가지|사람|달|이에|이야|이|가|은|는|을|를|에|의|와|과|요|입|씩|째|랑)"
)

_tens_pat = "|".join(sorted(_KO_NATIVE_TENS, key=len, reverse=True))
_ones_pat = "|".join(sorted(_KO_NATIVE_ONES, key=len, reverse=True))
_attr_pat = "|".join(_KO_NATIVE_ATTR)
_attr_tens_pat = "|".join(_KO_NATIVE_ATTR_TENS)

# Group layout: (1)tens+(2)ones/attr | (3)attr-tens | (4)tens-only | (5)ones-only | (6)attr-only
_KO_NATIVE_RE = re.compile(
    rf"(?<![가-힣])(?:({_tens_pat}) ?({_ones_pat}|{_attr_pat})|({_attr_tens_pat})|({_tens_pat})|({_ones_pat})|({_attr_pat}))"
)


def _convert_ko_native(m: re.Match) -> str:
    after = m.string[m.end() :]
    tens, ones, attr_tens, tens_only, ones_only, attr_only = m.groups()
    uses_attr = (ones in _KO_NATIVE_ATTR) or attr_tens or attr_only
    if uses_attr:
        if not _KO_NATIVE_COUNTER_RE.match(after):
            return m.group()
    elif after and _HANGUL_RE.match(after[0]) and not _KO_NATIVE_FOLLOW_RE.match(after):
        return m.group()
    if tens:
        return str(_KO_NATIVE_TENS[tens] + {**_KO_NATIVE_ONES, **_KO_NATIVE_ATTR}[ones])
    if attr_tens:
        return str(_KO_NATIVE_ATTR_TENS[attr_tens])
    if tens_only:
        return str(_KO_NATIVE_TENS[tens_only])
    if ones_only:
        return str(_KO_NATIVE_ONES[ones_only])
    return str(_KO_NATIVE_ATTR[attr_only])


# A number followed by a space and a counter is written either way
# (20 살 / 20살); join them so spacing does not count as an error.
_KO_DIGIT_COUNTER_SPACE_RE = re.compile(rf"(?<=\d) +(?=(?:{_KO_COUNTER_RE.pattern}))")
# Digit groups separated by spaces (010 1234) are one number in CER terms.
_KO_DIGIT_SPACE_RE = re.compile(r"(?<=\d) +(?=\d)")

_KO_PUNCT_RE = re.compile(r"[。，、；：？！…—～「」『』【】《》〈〉·。，]")
# Single-syllable words like 아/네/예/에 are excluded — they appear inside
# ordinary Korean words (e.g. 아홉=9) and would cause false stripping.
# Fillers are whole words only, so 음식, 다음, 영어, 열어 are kept.
_KO_FILLERS = re.compile(r"(?<![가-힣])(?:음+|어+|그러니까|그냥|뭐)(?![가-힣])")


class KoreanTextNormalizer(BaseTextNormalizer):
    """Normalize Korean text for WER/CER calculation.

    Handles both number systems:
    - Sino-Korean (한자어): positional, e.g. 삼백이십일 → 321, 1만 2천 → 12000.
      Trailing 일 after a number is the day counter (이십오일 → 25일).
      Digit strings read one by one (공일공 일이삼사) become digits.
    - Native Korean (고유어): 1–99, e.g. 스물하나 → 21, 열두 시 → 12시.

    Number syllables that are also words or surnames (이 "this", 일 "work",
    천/백 surnames, 만나다, 조금) are converted only in a number context;
    see ``_ko_convert_phrase`` and ``_convert_ko_native``.

    Pipeline:
    1. Remove markup and filler words
    2. Irregular months (유월 → 6월, 시월 → 10월)
    3. Spaced digit sequences (일 이 삼 → 123)
    4. Sino-Korean number phrases → Arabic digits
    5. Native Korean numbers → Arabic digits
    6. Strip Korean/CJK punctuation, lower-case ASCII
    7. Join a number and its counter (20 살 → 20살) and adjacent digit groups
    8. Collapse whitespace (word spaces are kept)
    """

    def __init__(self, ignore_patterns: str = ""):
        self._ignore_patterns = ignore_patterns or _KO_FILLERS.pattern

    def __call__(self, s: str) -> str:
        s = self._remove_markup(s)
        s = self._apply_ignore_patterns(s, self._ignore_patterns)
        s = self._collapse_whitespace(s)
        s = _KO_IRREGULAR_MONTH_RE.sub(lambda m: _KO_IRREGULAR_MONTHS[m.group(1)], s)
        s = _KO_DIGIT_SEQ_RE.sub(_convert_ko_digit_seq, s)
        s = _KO_PHRASE_RE.sub(_ko_split_groups, s)
        s = _KO_NATIVE_RE.sub(_convert_ko_native, s)
        s = _KO_PUNCT_RE.sub(" ", s)
        s = s.lower()
        s = self._collapse_whitespace(s)
        s = _KO_DIGIT_COUNTER_SPACE_RE.sub("", s)
        return _KO_DIGIT_SPACE_RE.sub("", s)
