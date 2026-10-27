"""WER text normalization utilities."""

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import regex
from jiwer import Compose, RemovePunctuation, Strip, ToLowerCase

from eva.utils.logging import get_logger
from eva.utils.wer_normalization.cjk import ChineseTextNormalizer, JapaneseTextNormalizer, KoreanTextNormalizer
from eva.utils.wer_normalization.engine import GenericTextNormalizer, LanguageConfig, make_char_folder
from eva.utils.wer_normalization.whisper_normalizer import BasicTextNormalizer

logger = get_logger(__name__)

_CONFIGS_DIR = Path(__file__).parent / "configs"
_DEFAULT_NORMALIZER = BasicTextNormalizer()


def _make_normalizer(language: str) -> BasicTextNormalizer | GenericTextNormalizer:
    """Return a normalizer for *language*, instantiated on demand.

    ``language`` must be an exact pipecat Language value (e.g. ``Language.FR.value``).
    Config files are named after those values verbatim (``fr-FR.json``, ``en.json``).
    """
    base = language.split("-")[0]
    if base == "ja":
        return JapaneseTextNormalizer()
    if base == "ko":
        return KoreanTextNormalizer()
    if base == "zh":
        return ChineseTextNormalizer()
    if (_CONFIGS_DIR / f"{language}.json").exists():
        return GenericTextNormalizer(LanguageConfig.load(language))
    logger.warning(f"No normalizer found for language {language}, using default normalizer")
    return _DEFAULT_NORMALIZER


# Basic transformations applied after Whisper normalization
BASIC_TRANSFORMATIONS = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        Strip(),
    ]
)

# Regex for apostrophes
RE_APOSTROPHES = re.compile(r"[''´`]")

# Splits commas only when they are NOT between digits, so number literals like
# "1,000" stay intact while comma-separated phrases ("nineteen ninety four,
# zero two, eleven", "Hello, World") become independent chunks. Without this,
# Whisper's process_words greedily concatenates consecutive number tokens
# across stripped punctuation (e.g. spelled-out dates become "19940211").
_RE_LIST_COMMA = re.compile(r"(?<!\d),(?!\d)")

# Drop leading zeros from multi-digit numbers so "02" (string output of
# spelled-out "zero two") matches "2" (Fraction-parsed output of digit "02").
_RE_LEADING_ZEROS = re.compile(r"\b0+(\d+)\b")

# Hyphen-joined word groups (e.g. "919-696-3901", "WZH-89B", "wishy-washy").
# Concatenate the digit-bearing ones so the digit form matches Whisper's spelled-out
# concatenation ("nine one nine ..." -> "9196963901"). Pure-letter compounds are
# left alone. ISO dates are special-cased: split into space-separated components so
# they match the spelled-out comma-separated form ("nineteen ninety four, ...").
_RE_HYPHEN_GROUPS = re.compile(r"\w+(?:-\w+)+")
_RE_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# When a user-simulator spells out an ID it often pronounces the literal "dash"
# between groups ("P R V dash S U R G dash zero zero four"), while STT writes
# the hyphen ("PRV-SURG-004"). Drop the spoken "dash" between alphanumeric
# tokens so both forms collapse the same way.
_RE_SPELLED_DASH = re.compile(r"(?<=\w)\s+dash\s+(?=\w)", re.IGNORECASE)


# Hyphen-joined runs of letters (incl. combining marks, so Devanagari vowel signs
# count: "डेटा-बेस"). Used only for languages with ``join_hyphenated_compounds``.
_RE_LETTER_COMPOUND = regex.compile(r"[\p{L}\p{M}]+(?:-[\p{L}\p{M}]+)+")


def _normalize_hyphen_groups(
    text: str,
    join_letter_compounds: bool = False,
    keep_boundary: Callable[[str], bool] | None = None,
) -> str:
    def repl(m: re.Match) -> str:
        s = m.group()
        if _RE_ISO_DATE.fullmatch(s):
            return s.replace("-", " ")
        if any(c.isdigit() for c in s):
            return s.replace("-", "")
        return s

    text = _RE_HYPHEN_GROUPS.sub(repl, text)
    if join_letter_compounds:
        # de/nl/hi: "Anmelde-Konto" == "Anmeldekonto", "ई-मेल" == "ईमेल".
        # The hyphen is kept next to a number word ("ABC-eins" must still become
        # "abc 1" to match "ABC-1") and next to a multi-letter acronym
        # ("HR-Fragen", "VPN-Passwort"), which STT often writes open ("HR Fragen").
        def keep(piece: str) -> bool:
            return (len(piece) > 1 and piece.isupper()) or (keep_boundary is not None and keep_boundary(piece))

        def join(m: regex.Match) -> str:
            parts = m.group().split("-")
            out = parts[0]
            for prev, cur in zip(parts, parts[1:]):
                # Reduplications ("बार-बार", "अभी-अभी") are as often written open as hyphenated: keep them split.
                out += ("-" if prev == cur or keep(prev) or keep(cur) else "") + cur
            return out

        text = _RE_LETTER_COMPOUND.sub(join, text)
    return text


_NUMBER_WORD_CHECKERS: dict[str, Callable[[str], bool]] = {}


def _number_word_checker(config: LanguageConfig) -> Callable[[str], bool]:
    """Return a predicate: is this token one of the language's number words? (cached per language code)"""
    if config.code in _NUMBER_WORD_CHECKERS:
        return _NUMBER_WORD_CHECKERS[config.code]
    fold = make_char_folder(config.char_folds)
    vocab = {
        fold(w.lower())
        for group in (config.zeros, config.cardinals, config.cardinal_variants, config.tens, config.scaling_units)
        for w in group
    }
    checker = _NUMBER_WORD_CHECKERS[config.code] = lambda token: fold(token.lower()) in vocab
    return checker


def _phrase_pattern(phrase: str) -> str:
    """Regex for a spoken-symbol phrase: spaces/apostrophes match any whitespace or apostrophe.

    Curly apostrophes are accepted here because ``normalize_apostrophes`` leaves them as-is.
    """
    parts = [re.escape(p) for p in re.split(r"[\s'’‘]+", phrase.strip()) if p]
    return r"(?:\s+|\s*['’‘]\s*)".join(parts)


@lru_cache(maxsize=64)
def _spoken_symbol_regex(phrases: tuple[str, ...]) -> re.Pattern | None:
    if not phrases:
        return None
    alts = "|".join(_phrase_pattern(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"(?<=\w)\s+({alts})\s+(?=\w)", re.IGNORECASE)


def _looks_like_code(token: str) -> bool:
    """A token from a spelled ID: has a digit, or is all-uppercase letters ("A", "ABC")."""
    token = token.strip(".,;:!?\"'()")
    return bool(token) and (any(c.isdigit() for c in token) or (token.isalpha() and token.isupper()))


def _replace_spoken_symbols(text: str, symbols: dict[str, str], code_only: dict[str, str]) -> str:
    """Replace spoken punctuation words between word tokens with the written symbol.

    "A B C trait d'union 1 2 3" -> "A B C-1 2 3", so it then normalizes like the
    STT form "ABC-123". Words in ``code_only`` are also ordinary words, so they
    are replaced only when a neighbouring token looks like part of a code.
    """

    def lookup(table: dict[str, str], spoken: str) -> str:
        key = re.sub(r"[\s'’‘]+", " ", spoken.lower())
        for phrase, symbol in table.items():
            if re.sub(r"[\s'’‘]+", " ", phrase.lower()) == key:
                return symbol
        return spoken  # unreachable: the regex only matches table phrases

    rx = _spoken_symbol_regex(tuple(symbols))
    if rx is not None:
        text = rx.sub(lambda m: lookup(symbols, m.group(1)), text)

    rx = _spoken_symbol_regex(tuple(code_only))
    if rx is not None:

        def repl(m: re.Match) -> str:
            before = m.string[: m.start()].split()
            after = m.string[m.end() :].split()
            if (before and _looks_like_code(before[-1])) or (after and _looks_like_code(after[0])):
                return lookup(code_only, m.group(1))
            return m.group(0)

        text = rx.sub(repl, text)
    return text


def normalize_apostrophes(text: str) -> str:
    """Normalize apostrophes in the text to a standard single quote."""
    return RE_APOSTROPHES.sub("'", text)


def convert_unicode_to_characters(text: str) -> str:
    r"""Convert unicode (\u00e9) to characters (é).

    Only applied when the text is pure ASCII — the round-trip via
    raw_unicode_escape/unicode-escape is only safe for ASCII strings containing
    \uXXXX escape sequences. Non-ASCII text (CJK, Arabic, accented Latin, …)
    is already in the correct form and must not be re-encoded.
    """
    if not text.isascii():
        return text
    return text.encode("raw_unicode_escape").decode("unicode-escape")


def collapse_single_letters(text: str) -> str:
    """Collapse sequences of 3 or more single letters separated by spaces. Such as a b c -> abc."""
    return re.sub(r"\b(?:[a-zA-Z] ){2,}[a-zA-Z]\b", lambda m: "".join(m.group(0).split()).upper(), text)


def remove_space_between_numbers_and_suffix(text: str) -> str:
    """Remove space between numbers and suffixes like 'th', 'nd', 'st'."""
    return re.sub(r"(?<=\d)\s+(?=(?:st|nd|rd|th)\b)", "", text)


def normalize_text(text: str, language: str = "en") -> str:
    """Normalize text based on language.

    Args:
        text: Input text to normalize
        language: Language code (default: "en")

    Returns:
        Normalized text string

    Pipeline:
        1. Convert unicode sequences to characters
        2. Normalize apostrophes to standard single quote
        3. Collapse single letters (e.g., "a b c" -> "ABC")
        4. Concatenate hyphen-joined groups containing digits ("919-696-3901"
           -> "9196963901", "WZH-89B" -> "WZH89B") so the digit form matches
           Whisper's spelled-out concatenation. ISO dates are split instead.
        5. Split on list-style commas (preserves "1,000" but breaks
           "nineteen ninety four, zero two, eleven" into chunks so number
           tokens don't concatenate across the comma)
        6. Apply Whisper normalizer + basic transformations to each chunk
           (Whisper also normalizes spelled-out numbers, e.g. "twenty two" -> "22")
        7. Strip leading zeros so "zero two" -> "02" -> "2" matches digit "02" -> "2"
        8. Remove space between numbers and suffixes (e.g., "3 rd" -> "3rd")
    """
    try:
        normalizer = _make_normalizer(language)
        config = getattr(normalizer, "config", None)
        text = convert_unicode_to_characters(text)
        text = normalize_apostrophes(text)
        text = collapse_single_letters(text)
        text = _RE_SPELLED_DASH.sub(" ", text)
        if config is not None:
            text = _replace_spoken_symbols(text, config.spoken_symbols, config.spoken_symbols_code_only)
        if config is not None and config.join_hyphenated_compounds:
            text = _normalize_hyphen_groups(text, True, _number_word_checker(config))
        else:
            text = _normalize_hyphen_groups(text)
        chunks = _RE_LIST_COMMA.split(text)
        normalized_chunks = [BASIC_TRANSFORMATIONS([normalizer(c)])[0] for c in chunks]
        text = " ".join(c for c in normalized_chunks if c)
        text = _RE_LEADING_ZEROS.sub(r"\1", text)
        text = remove_space_between_numbers_and_suffix(text)
    except Exception:
        logger.exception(f"Error normalizing {text}.")
    return text
