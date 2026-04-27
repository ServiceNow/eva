"""WER text normalization utilities."""

import re

from jiwer import Compose, RemovePunctuation, Strip, ToLowerCase

from eva.utils.logging import get_logger
from eva.utils.wer_normalization.normalizers import JapaneseTextNormalizer
from eva.utils.wer_normalization.whisper_normalizer.basic import BasicTextNormalizer
from eva.utils.wer_normalization.whisper_normalizer.english import EnglishTextNormalizer

logger = get_logger(__name__)

# Normalizers per language
NORMALIZERS = {"en": EnglishTextNormalizer(), "ja": JapaneseTextNormalizer()}
DEFAULT_NORMALIZER = BasicTextNormalizer()

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


def _normalize_hyphen_groups(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group()
        if _RE_ISO_DATE.fullmatch(s):
            return s.replace("-", " ")
        if any(c.isdigit() for c in s):
            return s.replace("-", "")
        return s

    return _RE_HYPHEN_GROUPS.sub(repl, text)


def normalize_apostrophes(text: str) -> str:
    """Normalize apostrophes in the text to a standard single quote."""
    return RE_APOSTROPHES.sub("'", text)


def convert_unicode_to_characters(text: str) -> str:
    r"""Convert unicode (\u00e9) to characters (é)."""
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
        normalizer = NORMALIZERS.get(language, DEFAULT_NORMALIZER)
        text = convert_unicode_to_characters(text)
        text = normalize_apostrophes(text)
        text = collapse_single_letters(text)
        text = _RE_SPELLED_DASH.sub(" ", text)
        text = _normalize_hyphen_groups(text)
        chunks = _RE_LIST_COMMA.split(text)
        normalized_chunks = [BASIC_TRANSFORMATIONS([normalizer(c)])[0] for c in chunks]
        text = " ".join(c for c in normalized_chunks if c)
        text = _RE_LEADING_ZEROS.sub(r"\1", text)
        text = remove_space_between_numbers_and_suffix(text)
    except Exception:
        logger.exception(f"Error normalizing {text}.")
    return text
