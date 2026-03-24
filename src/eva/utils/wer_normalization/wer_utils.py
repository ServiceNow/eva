"""WER text normalization utilities."""

import re

import inflect
from jiwer import Compose, RemovePunctuation, Strip, ToLowerCase

from eva.utils.logging import get_logger
from eva.utils.wer_normalization.normalizers import JapaneseTextNormalizer
from eva.utils.wer_normalization.whisper_normalizer.basic import BasicTextNormalizer
from eva.utils.wer_normalization.whisper_normalizer.english import EnglishTextNormalizer

logger = get_logger(__name__)

_inflect_engine = inflect.engine()

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


def normalize_apostrophes(text: str) -> str:
    """Normalize apostrophes in the text to a standard single quote."""
    return RE_APOSTROPHES.sub("'", text)


def convert_unicode_to_characters(text: str) -> str:
    r"""Convert unicode (\u00e9) to characters (é)."""
    return text.encode("raw_unicode_escape").decode("unicode-escape")


def convert_digits_to_words(text: str, language: str):
    """Convert numbers to words (e.g., "3" to "three").

    Only English is supported. For non-English languages, returns text unchanged.
    """
    if language != "en":
        return text

    def replace_num_with_words_and_space(m) -> str:
        """Add space before and after the word if needed, to avoid 1B9 becoming onebnine."""
        try:
            word = _inflect_engine.number_to_words(int(m.group()))
        except Exception as e:  # We have seen OverflowError in some cases
            logger.error(f"Error converting digits to words: {e}, text: {m}")
            return m.group()

        start = m.start()
        end = m.end()
        s = m.string

        # Add space before if not at start and previous char is not space or punctuation
        if start > 0 and s[start - 1] not in " \t\n.,;:!?":
            word = " " + word
        # Add space after if not at end and next char is not space or punctuation
        if end < len(s) and s[end] not in " \t\n.,;:!?":
            word += " "
        return word

    return re.sub(r"\d+", replace_num_with_words_and_space, text)


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
        2. Convert digits to words (e.g., "3" -> "three")
        3. Normalize apostrophes to standard single quote
        4. Collapse single letters (e.g., "a b c" -> "ABC")
        5. Apply Whisper normalizer (language-specific)
        6. Apply basic transformations (lowercase, remove punctuation, strip)
        7. Remove space between numbers and suffixes (e.g., "3 rd" -> "3rd")
    """
    try:
        normalizer = NORMALIZERS.get(language, DEFAULT_NORMALIZER)
        text = convert_unicode_to_characters(text)
        text = convert_digits_to_words(text, language)
        text = normalize_apostrophes(text)
        text = collapse_single_letters(text)
        text = BASIC_TRANSFORMATIONS([normalizer(text)])[0]
        text = remove_space_between_numbers_and_suffix(text)
    except Exception:
        logger.exception(f"Error normalizing {text}.")
    return text
