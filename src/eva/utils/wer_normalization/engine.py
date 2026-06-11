"""Data-driven number/text normalizer engine for WER.

A single engine that consumes per-language JSON configs and reproduces the
behaviour of the original Whisper-style English and French normalizers.
Add a new language by dropping a JSON file into ``configs/`` — no Python
needed for languages that fit the supported feature set (base-10 with
optional vigesimal augmentation, decimal-comma or decimal-point, etc.).

Languages with fundamentally different number systems (CJK, Korean) should
keep dedicated normalizer classes; this engine is for alphabetic languages.
"""

import json
import re
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from re import Match
from typing import Any

from more_itertools import windowed
from pydantic import BaseModel, Field

from eva.utils.wer_normalization.whisper_normalizer.basic import (
    remove_symbols_and_diacritics,
    remove_symbols_keep_marks,
)

CONFIGS_DIR = Path(__file__).parent / "configs"


class LanguageConfig(BaseModel):
    """Per-language normalization config.

    Pure data plus a small set of structural flags. See ``configs/en.json`` /
    ``configs/fr.json`` for concrete examples.
    """

    code: str

    # --- Number word maps (suffixed = plural/ordinal merged) -------------
    zeros: list[str]
    cardinals: dict[str, int]
    cardinal_variants: dict[str, int] = Field(default_factory=dict)
    cardinals_suffixed: dict[str, tuple[int, str]] = Field(default_factory=dict)
    tens: dict[str, int] = Field(default_factory=dict)
    tens_suffixed: dict[str, tuple[int, str]] = Field(default_factory=dict)
    scaling_units: dict[str, int] = Field(default_factory=dict)
    scaling_units_suffixed: dict[str, tuple[int, str]] = Field(default_factory=dict)

    # --- Symbols / currency / percent ------------------------------------
    preceding_prefixers: dict[str, str] = Field(default_factory=dict)
    following_prefixers: dict[str, str] = Field(default_factory=dict)
    # Each value is either a literal symbol or a {next_word: symbol} dict
    # for two-word forms like "per cent".
    suffixers: dict[str, Any] = Field(default_factory=dict)

    # --- Conjunction / decimal / repeat words ----------------------------
    conjunction_word: str | None = None  # "and" / "et"
    conjunction_ignore_prev: list[str] = Field(default_factory=list)
    repeat_words: dict[str, int] = Field(default_factory=dict)  # {"double": 2}
    decimal_word: str | None = None  # "point" / "virgule"

    # --- Structural number rules -----------------------------------------
    # Vigesimal augmentation: when one of ``trigger_words`` appears and the
    # accumulated value's residual mod 100 is in ``residuals``, fold the
    # accumulator with the multiplier (French: vingt after 4-9 -> *20).
    vigesimal_trigger_words: list[str] = Field(default_factory=list)
    vigesimal_multiplier: int = 20
    vigesimal_residuals: list[int] = Field(default_factory=list)

    # Which ``value % 100`` residuals allow ones>=10 (e.g. "dix") to be
    # added (rather than concatenated as a string). EN: [0]; FR: [0, 60, 80].
    additive_teen_residuals: list[int] = Field(default_factory=lambda: [0])

    # Whether ones combination also fires when prev was a ones word and
    # ``value`` is still int (English needs this for "one oh one" form).
    cardinal_continuation_on_prev_cardinal: bool = False

    # --- Preprocessing ---------------------------------------------------
    half_pattern: str | None = None  # e.g. r"\band\s+a\s+half\b"
    half_replacement: str | None = None  # e.g. "point five"
    split_hyphenated_numbers: bool = False
    # Languages where the units word precedes the tens word, optionally glued
    # by ``conjunction_word`` (DE "einundzwanzig", AR "wahid wa-'ishrun").
    # When true, a preprocessor swaps the order before the state machine runs.
    reversed_units: bool = False
    letter_class: str = "a-z"  # used in number/letter boundary regex
    ordinal_suffix_pattern: str = ""  # e.g. r"(?:st|nd|rd|th|s)"

    # --- Postprocessing --------------------------------------------------
    one_word: str | None = None  # postprocess "1" -> word
    one_plural_suffix: str = ""  # "s" for English "ones"
    cents_connector: str | None = None  # word optionally between currency and cents

    # --- Text-level (outer) normalizer -----------------------------------
    ignore_patterns: str = ""
    replacers: dict[str, str] = Field(default_factory=dict)
    strip_space_before_apostrophe: bool = False
    thousand_separators: list[str] = Field(default_factory=list)  # any of: comma|dot|space
    decimal_separator: str = "dot"  # dot | comma
    spelling_map_path: str | None = None
    # For scripts where combining marks are linguistically essential
    # (Devanagari, Bengali, Tamil, Arabic, Hebrew, Thai). When true, the
    # outer text normalizer uses ``remove_symbols_keep_marks`` instead of
    # ``remove_symbols_and_diacritics`` so vowel signs survive.
    preserve_combining_marks: bool = False

    @classmethod
    def load(cls, language: str) -> "LanguageConfig":
        path = CONFIGS_DIR / f"{language}.json"
        with open(path) as f:
            return cls.model_validate(json.load(f))


class GenericNumberNormalizer:
    """Convert spelled-out numbers into arabic digits, config-driven.

    Algorithm is the Whisper number-normalizer state machine with the
    language-specific quirks (vigesimal, conjunction, ones-string continuation,
    decimal word) parametrized via ``LanguageConfig``.
    """

    def __init__(self, config: LanguageConfig):
        self.config = config
        # Strip diacritics from vocab keys so lookups match the text after
        # GenericTextNormalizer.__call__ runs remove_symbols_and_diacritics on
        # the input. Without this, vocab entries like "zwölf"/"dreißig"/"fünf"
        # never match because the input has already been folded to ASCII.
        _strip_fn = remove_symbols_keep_marks if config.preserve_combining_marks else remove_symbols_and_diacritics
        strip = lambda w: _strip_fn(w, keep="")  # noqa: E731
        _strip_dict = lambda d: {strip(k): v for k, v in d.items()}  # noqa: E731
        # For suffixed dicts the value is (int, suffix_str); strip the suffix too,
        # otherwise emitted tokens (e.g. "2ème") won't match digit-form outputs
        # which have been accent-stripped to "2eme" upstream.
        _strip_suffixed = lambda d: {strip(k): (v[0], strip(v[1])) for k, v in d.items()}  # noqa: E731

        self.zeros = {strip(w) for w in config.zeros}
        self.cardinals = {**_strip_dict(config.cardinals), **_strip_dict(config.cardinal_variants)}
        self.cardinals_suffixed = _strip_suffixed(config.cardinals_suffixed)
        self.tens = _strip_dict(config.tens)
        self.tens_suffixed = _strip_suffixed(config.tens_suffixed)
        self.scaling_units = _strip_dict(config.scaling_units)
        self.scaling_units_suffixed = _strip_suffixed(config.scaling_units_suffixed)
        self.preceding_prefixers = _strip_dict(config.preceding_prefixers)
        self.following_prefixers = _strip_dict(config.following_prefixers)
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {strip(k): v for k, v in config.suffixers.items()}
        self.repeat_words = _strip_dict(config.repeat_words)
        # Trigger words may carry accents (e.g. "vingtième"); the tokens reaching
        # process_words are already accent-stripped, so strip these to match.
        self.vigesimal_trigger_words = {strip(w) for w in config.vigesimal_trigger_words}
        self.decimals = {*self.cardinals, *self.tens, *self.zeros}

        self.specials: set[str] = set()
        if config.conjunction_word:
            self.specials.add(config.conjunction_word)
        if config.decimal_word:
            self.specials.add(config.decimal_word)
        self.specials |= set(config.repeat_words)

        self.words = (
            set(self.zeros)
            | set(self.cardinals)
            | set(self.cardinals_suffixed)
            | set(self.tens)
            | set(self.tens_suffixed)
            | set(self.scaling_units)
            | set(self.scaling_units_suffixed)
            | set(self.preceding_prefixers)
            | set(self.following_prefixers)
            | set(self.suffixers)
            | self.specials
        )

        self._reversed_units_glued: re.Pattern | None = None
        self._reversed_units_spaced: re.Pattern | None = None
        self._glued_splitters: list[re.Pattern] = []
        if config.reversed_units and self.cardinals:
            cardinals_pat = "|".join(re.escape(w) for w in sorted(self.cardinals, key=len, reverse=True))
            if config.conjunction_word and self.tens:
                conj_stripped = remove_symbols_and_diacritics(config.conjunction_word, keep="")
                tens_pat = "|".join(re.escape(w) for w in sorted(self.tens, key=len, reverse=True))
                conj = re.escape(conj_stripped)
                self._reversed_units_glued = re.compile(rf"\b({cardinals_pat}){conj}({tens_pat})\b", re.IGNORECASE)
                self._reversed_units_spaced = re.compile(
                    rf"\b({cardinals_pat})\s+{conj}\s+({tens_pat})\b", re.IGNORECASE
                )
            if self.scaling_units:
                scale_pat = "|".join(re.escape(w) for w in sorted(self.scaling_units, key=len, reverse=True))
                tens_pat = "|".join(re.escape(w) for w in sorted(self.tens, key=len, reverse=True)) if self.tens else ""
                # Split glued <ones><multiplier> compounds ("zweihundert" → "zwei hundert").
                # No trailing \b: in mega-compounds the next char is a letter,
                # not a boundary. Iteration in preprocess() converges.
                self._glued_splitters.append(re.compile(rf"\b({cardinals_pat})({scale_pat})", re.IGNORECASE))
                # Split glued <multiplier><ones|tens> compounds
                # ("hunderteinundzwanzig" → "hundert einundzwanzig",
                # "tausendzwanzig" → "tausend zwanzig").
                after_alts = cardinals_pat + (f"|{tens_pat}" if tens_pat else "")
                self._glued_splitters.append(re.compile(rf"\b({scale_pat})({after_alts})", re.IGNORECASE))

    def _cardinal_should_concat_as_str(self, prev: str | None, value: int | str | None) -> bool:
        if isinstance(value, str):
            return True
        if self.config.cardinal_continuation_on_prev_cardinal and prev in self.cardinals:
            return True
        return False

    def process_words(self, words: list[str]) -> Iterator[str]:
        prefix: str | None = None
        value: str | int | None = None
        skip = False

        conjunction = self.config.conjunction_word
        decimal_word = self.config.decimal_word
        conjunction_ignore_prev_sets = []
        for cat in self.config.conjunction_ignore_prev:
            conjunction_ignore_prev_sets.append(getattr(self, cat))

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: str | int) -> str:
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if not words:
            return

        for prev, current, nxt in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = nxt is not None and re.match(r"^\d+(\.\d+)?$", nxt)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current

            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                f = to_fraction(current_without_prefix)
                if f is not None:
                    if value is not None:
                        if isinstance(value, str) and value.endswith("."):
                            value = str(value) + str(current)
                            continue
                        else:
                            yield output(value)
                    prefix = current[0] if has_prefix else prefix
                    value = f.numerator if f.denominator == 1 else current_without_prefix
            elif current not in self.words:
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.cardinals:
                n = self.cardinals[current]
                if value is None:
                    value = n
                elif self._cardinal_should_concat_as_str(prev, value):
                    if prev in self.tens and n < 10:
                        assert isinstance(value, str) and value[-1] == "0"
                        value = value[:-1] + str(n)
                    else:
                        value = str(value) + str(n)
                elif n < 10:
                    if value % 10 == 0:
                        value += n
                    else:
                        value = str(value) + str(n)
                else:  # ones >= 10 (teens / dix-seize)
                    if value % 100 in self.config.additive_teen_residuals:
                        value += n
                    else:
                        value = str(value) + str(n)
            elif current in self.cardinals_suffixed:
                n, suffix = self.cardinals_suffixed[current]
                if value is None:
                    yield output(str(n) + suffix)
                elif self._cardinal_should_concat_as_str(prev, value):
                    if prev in self.tens and n < 10:
                        assert isinstance(value, str) and value[-1] == "0"
                        yield output(value[:-1] + str(n) + suffix)
                    else:
                        yield output(str(value) + str(n) + suffix)
                elif n < 10:
                    if value % 10 == 0:
                        yield output(str(value + n) + suffix)
                    else:
                        yield output(str(value) + str(n) + suffix)
                else:
                    if value % 100 in self.config.additive_teen_residuals:
                        yield output(str(value + n) + suffix)
                    else:
                        yield output(str(value) + str(n) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if (
                    current in self.vigesimal_trigger_words
                    and isinstance(value, int)
                    and value % 100 in self.config.vigesimal_residuals
                ):
                    base = (value // 100) * 100
                    digit = value % 100
                    value = base + digit * self.config.vigesimal_multiplier
                elif value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                tens, suffix = self.tens_suffixed[current]
                if (
                    current in self.vigesimal_trigger_words
                    and isinstance(value, int)
                    and value % 100 in self.config.vigesimal_residuals
                ):
                    base = (value // 100) * 100
                    digit = value % 100
                    yield output(str(base + digit * self.config.vigesimal_multiplier) + suffix)
                elif value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.scaling_units:
                scale = self.scaling_units[current]
                if value is None:
                    value = scale
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * scale if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = scale
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * scale
            elif current in self.scaling_units_suffixed:
                scale, suffix = self.scaling_units_suffixed[current]
                if value is None:
                    yield output(str(scale) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * scale if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(scale) + suffix)
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * scale
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                if value is not None:
                    yield output(value)
                if nxt in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if nxt in suffix:
                            yield output(str(value) + suffix[nxt])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if nxt not in self.words and not next_is_numeric:
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == conjunction:
                    if not any(prev in s for s in conjunction_ignore_prev_sets):
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current in self.repeat_words:
                    if nxt in self.cardinals or nxt in self.zeros:
                        repeats = self.repeat_words[current]
                        cardinal_val = self.cardinals.get(nxt, 0)
                        value = str(value or "") + str(cardinal_val) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == decimal_word:
                    if nxt in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    raise ValueError(f"Unexpected token: {current}")
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str) -> str:
        if self.config.half_pattern and self.config.half_replacement:
            results = []
            segments = re.split(self.config.half_pattern, s)
            for i, segment in enumerate(segments):
                if not segment.strip():
                    continue
                if i == len(segments) - 1:
                    results.append(segment)
                else:
                    results.append(segment)
                    last_word = segment.rsplit(maxsplit=2)[-1]
                    if last_word in self.decimals or last_word in self.scaling_units:
                        results.append(self.config.half_replacement)
                    else:
                        # keep original literal (reconstruct rough form)
                        results.append(self._half_literal())
            s = " ".join(results)

        if self._glued_splitters:
            for _ in range(10):
                new = s
                for pat in self._glued_splitters:
                    new = pat.sub(lambda m: f"{m.group(1)} {m.group(2)}", new)
                if new == s:
                    break
                s = new
        if self._reversed_units_glued is not None:
            s = self._reversed_units_glued.sub(lambda m: f"{m.group(2)} {m.group(1)}", s)
        if self._reversed_units_spaced is not None:
            s = self._reversed_units_spaced.sub(lambda m: f"{m.group(2)} {m.group(1)}", s)

        if self.config.split_hyphenated_numbers:

            def split_number_hyphens(m: Match) -> str:
                token = m.group(0)
                parts = token.split("-")
                if all(p.lower() in self.words or p.lower() in self.zeros for p in parts):
                    return " ".join(parts)
                return token

            s = re.sub(r"\b[\w]+([-][\w]+)+\b", split_number_hyphens, s)

        letter_class = self.config.letter_class
        s = re.sub(rf"([{letter_class}])([0-9])", r"\1 \2", s)
        s = re.sub(rf"([0-9])([{letter_class}])", r"\1 \2", s)

        if self.config.ordinal_suffix_pattern:
            s = re.sub(rf"([0-9])\s+({self.config.ordinal_suffix_pattern})\b", r"\1\2", s)

        return s

    def _half_literal(self) -> str:
        """Best-effort literal to put back when a 'half' pattern fired but didn't apply."""
        # Used only when we split on the half pattern but the prior word isn't a number.
        # Mirrors the original behaviour of falling back to a canonical literal.
        return "and a half" if self.config.conjunction_word == "and" else "et demi"

    def postprocess(self, s: str) -> str:
        connector = f"(?:{self.config.cents_connector} )?" if self.config.cents_connector else ""

        def combine_cents(m: Match) -> str:
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match) -> str:
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        s = re.sub(rf"([€£$])([0-9]+) {connector}¢([0-9]{{1,2}})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        if self.config.one_word:
            suffix = self.config.one_plural_suffix
            if suffix:
                s = re.sub(rf"\b1({suffix}?)\b", rf"{self.config.one_word}\1", s)
            else:
                s = re.sub(r"\b1\b", self.config.one_word, s)

        return s

    def __call__(self, s: str) -> str:
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        # Re-apply digit/letter boundary spacing: state machine emits "2eme"
        # glued, while a digit-form input "2eme" was already split to "2 eme"
        # by preprocess(). Apply here for output symmetry.
        letter_class = self.config.letter_class
        s = re.sub(rf"([{letter_class}])([0-9])", r"\1 \2", s)
        s = re.sub(rf"([0-9])([{letter_class}])", r"\1 \2", s)
        s = self.postprocess(s)
        return s


_RE_ANGLE_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]")
_RE_PARENS = re.compile(r"\(([^)]+?)\)")
_RE_WHITESPACE = re.compile(r"\s+")


class BaseTextNormalizer:
    """Shared structural scaffold for all language text normalizers.

    Provides markup removal, optional ignore-pattern filtering, and final
    whitespace collapsing. Subclasses override ``__call__`` and call these
    helpers as needed.
    """

    def _remove_markup(self, s: str) -> str:
        """Strip <tag>, [tag], and (parenthetical) spans."""
        s = _RE_ANGLE_BRACKETS.sub("", s)
        s = _RE_PARENS.sub("", s)
        return s

    def _apply_ignore_patterns(self, s: str, pattern: str) -> str:
        return re.sub(pattern, "", s) if pattern else s

    def _collapse_whitespace(self, s: str) -> str:
        return _RE_WHITESPACE.sub(" ", s).strip()

    def __call__(self, s: str) -> str:
        raise NotImplementedError


class GenericTextNormalizer(BaseTextNormalizer):
    """Config-driven text normalizer for alphabetic languages."""

    def __init__(self, config: LanguageConfig):
        self.config = config
        self.standardize_numbers = GenericNumberNormalizer(config)
        self.spelling_map: dict[str, str] = {}
        if config.spelling_map_path:
            with open(CONFIGS_DIR / config.spelling_map_path) as f:
                self.spelling_map = json.load(f)

    def __call__(self, s: str) -> str:
        s = s.lower()
        s = self._remove_markup(s)
        s = self._apply_ignore_patterns(s, self.config.ignore_patterns)
        if self.config.strip_space_before_apostrophe:
            s = re.sub(r"\s+(?=')", "", s)

        for pattern, replacement in self.config.replacers.items():
            s = re.sub(pattern, replacement, s)

        if "comma" in self.config.thousand_separators:
            s = re.sub(r"(\d),(\d)", r"\1\2", s)
        if "dot" in self.config.thousand_separators:
            s = re.sub(r"(\d)\.(\d{3})(?!\d)", r"\1\2", s)
        if "space" in self.config.thousand_separators:
            s = re.sub(r"(\d) (\d{3})(?!\d)", r"\1\2", s)

        if self.config.decimal_separator == "comma":
            s = re.sub(r"(\d),(\d)", r"\1.\2", s)

        s = re.sub(r"\.(?!\d)", " ", s)
        strip_fn = remove_symbols_keep_marks if self.config.preserve_combining_marks else remove_symbols_and_diacritics
        s = strip_fn(s, keep=".%$¢€£")

        s = self.standardize_numbers(s)

        if self.spelling_map:
            s = " ".join(self.spelling_map.get(w, w) for w in s.split())

        s = re.sub(r"(\d+)\s*([$¢€£])", r"\2\1", s)
        s = re.sub(r"(?<=[$¢€£])\s+(?=\d)", "", s)
        s = re.sub(r"[.$¢€£](?!\d)", " ", s)
        s = re.sub(r"(?<!\d)%", " ", s)
        return self._collapse_whitespace(s)
