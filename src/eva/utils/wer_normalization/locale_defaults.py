"""BCP-47 → locale defaults for WER normalizer configs.

Tables for fields that are deterministic per-locale and should NOT be
LLM-generated when onboarding a new language: number-formatting separators,
the regex letter class used for letter/digit boundary detection, etc.

Keep these small and conservative. When a language isn't listed, fall back
to the safe defaults (dot decimal, no thousand separators, Latin letter class).
"""

# Script-derived letter class for the engine's letter/digit boundary regex.
# Keys are ISO 15924 script categories; values are regex character classes
# (kept ASCII-safe where possible — engine compiles these without re.UNICODE).
LETTER_CLASS_BY_SCRIPT: dict[str, str] = {
    "Latin": "a-zà-öø-ÿœæ",  # Western + most diacritics
    "Cyrillic": "a-zа-яё",
    "Greek": "a-zα-ωάέήίόύώϊϋΐΰ",
    "Arabic": "a-zء-ي",
    "Hebrew": "a-zא-ת",
    "Devanagari": "a-zऀ-ॿ",
    "Bengali": "a-zঀ-৿",
    "Gurmukhi": "a-zਁ-ੴ",
    "Gujarati": "a-zઁ-૱",
    "Oriya": "a-zଁ-ୱ",
    "Tamil": "a-zஂ-௿",
    "Telugu": "a-zఁ-౿",
    "Kannada": "a-zಁ-ೲ",
    "Malayalam": "a-zഁ-ൿ",
    "Sinhala": "a-zඁ-෴",
    "Thai": "a-zก-๛",
    "Lao": "a-zກ-ໝ",
    "Khmer": "a-zក-៰",
    "Myanmar": "a-zက-႟",
    "Tibetan": "a-zༀ-࿿",
}

# Decimal/thousand separator conventions per BCP-47 base language tag.
# `decimal_separator`: "dot" | "comma"
# `thousand_separators`: subset of {"comma", "dot", "space"}
SEPARATORS_BY_LANG: dict[str, dict] = {
    "en": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
    "fr": {"decimal_separator": "comma", "thousand_separators": ["dot", "space"]},
    "de": {"decimal_separator": "comma", "thousand_separators": ["dot", "space"]},
    "nl": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "es": {"decimal_separator": "comma", "thousand_separators": ["dot", "space"]},
    "it": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "pt": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "ro": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "ru": {"decimal_separator": "comma", "thousand_separators": ["space"]},
    "pl": {"decimal_separator": "comma", "thousand_separators": ["space"]},
    "tr": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "hu": {"decimal_separator": "comma", "thousand_separators": ["space"]},
    "fi": {"decimal_separator": "comma", "thousand_separators": ["space"]},
    "ar": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
    "he": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
    "hi": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
    "bn": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
    "vi": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "id": {"decimal_separator": "comma", "thousand_separators": ["dot"]},
    "ms": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
    "sw": {"decimal_separator": "dot", "thousand_separators": ["comma"]},
}

# BCP-47 base language → script. Only entries we have a letter class for.
SCRIPT_BY_LANG: dict[str, str] = {
    "en": "Latin",
    "fr": "Latin",
    "de": "Latin",
    "nl": "Latin",
    "es": "Latin",
    "it": "Latin",
    "pt": "Latin",
    "ro": "Latin",
    "tr": "Latin",
    "hu": "Latin",
    "fi": "Latin",
    "vi": "Latin",
    "id": "Latin",
    "ms": "Latin",
    "sw": "Latin",
    "pl": "Latin",
    "ru": "Cyrillic",
    "uk": "Cyrillic",
    "bg": "Cyrillic",
    "sr": "Cyrillic",
    "el": "Greek",
    "ar": "Arabic",
    "fa": "Arabic",
    "ur": "Arabic",
    "he": "Hebrew",
    "hi": "Devanagari",
    "mr": "Devanagari",
    "ne": "Devanagari",
    "sa": "Devanagari",
    "bn": "Bengali",
    "as": "Bengali",
    "pa": "Gurmukhi",
    "gu": "Gujarati",
    "or": "Oriya",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "si": "Sinhala",
    "th": "Thai",
    "lo": "Lao",
    "km": "Khmer",
    "my": "Myanmar",
    "bo": "Tibetan",
}


def base_lang(language: str) -> str:
    """Return BCP-47 primary subtag in lowercase: 'es-MX' → 'es'."""
    return language.split("-")[0].lower()


# Scripts where combining marks are linguistically essential (vowel signs,
# virama, nukta, etc.). Languages in these scripts must use the mark-preserving
# stripper or whole words get shredded into consonant fragments.
SCRIPTS_PRESERVE_MARKS: set[str] = {
    # Brahmic family (combining vowel signs + virama are essential)
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Oriya",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Sinhala",
    # SE Asian Brahmic-derived
    "Thai",
    "Lao",
    "Khmer",
    "Myanmar",
    # Other complex scripts with phonemic combining marks
    "Tibetan",
    "Arabic",
    "Hebrew",
}


def locale_defaults(language: str) -> dict:
    """Return injectable LanguageConfig defaults for ``language``.

    Safe fallback for unknown languages: Latin letter class, dot decimal,
    no thousand separators. The caller should merge these into the LLM
    output before validation.
    """
    base = base_lang(language)
    script = SCRIPT_BY_LANG.get(base, "Latin")
    seps = SEPARATORS_BY_LANG.get(base, {"decimal_separator": "dot", "thousand_separators": []})
    return {
        "letter_class": LETTER_CLASS_BY_SCRIPT.get(script, "a-z"),
        "preserve_combining_marks": script in SCRIPTS_PRESERVE_MARKS,
        **seps,
    }
