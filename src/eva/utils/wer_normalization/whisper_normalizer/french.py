# ruff: noqa
import re
from fractions import Fraction
from collections.abc import Iterator
from typing import Match

from more_itertools import windowed

from .basic import remove_symbols_and_diacritics


class FrenchNumberNormalizer:
    """Convert any French spelled-out numbers into arabic numbers, while handling:

    - remove any spaces/periods used as thousand separators
    - handle French decimal comma (virgule)
    - handle vigesimal forms: soixante-dix (70), quatre-vingts (80), quatre-vingt-dix (90)
    - spell out currency symbols after the number. e.g. `20 millions d'euros` -> `20000000 euros`
    - handle ordinal suffixes: `premier`, `deuxième`, etc.
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"zéro", "zero"}
        self.ones = {
            name: i
            for i, name in enumerate(
                [
                    "un",
                    "deux",
                    "trois",
                    "quatre",
                    "cinq",
                    "six",
                    "sept",
                    "huit",
                    "neuf",
                    "dix",
                    "onze",
                    "douze",
                    "treize",
                    "quatorze",
                    "quinze",
                    "seize",
                ],
                start=1,
            )
        }
        # "une" is the feminine form of "un"
        self.ones["une"] = 1

        self.ones_ordinal = {
            "premier": (1, "er"),
            "première": (1, "ère"),
            "second": (2, "nd"),
            "seconde": (2, "nde"),
            **{
                name + ("ième" if name.endswith("e") else "ième"): (value, "ème")
                for name, value in self.ones.items()
                if name not in ("un", "une")
            },
        }
        # fix irregular ordinals
        self.ones_ordinal["cinquième"] = (5, "ème")
        self.ones_ordinal["neuvième"] = (9, "ème")
        self.ones_ordinal["unième"] = (1, "ème")  # as in vingt et unième

        self.ones_suffixed = {**self.ones_ordinal}

        self.tens = {
            "vingt": 20,
            "vingts": 20,
            "trente": 30,
            "quarante": 40,
            "cinquante": 50,
            "soixante": 60,
        }
        self.tens_ordinal = {
            "vingtième": (20, "ème"),
            "trentième": (30, "ème"),
            "quarantième": (40, "ème"),
            "cinquantième": (50, "ème"),
            "soixantième": (60, "ème"),
        }
        self.tens_suffixed = {**self.tens_ordinal}

        self.multipliers = {
            "cent": 100,
            "cents": 100,
            "mille": 1_000,
            "million": 1_000_000,
            "millions": 1_000_000,
            "milliard": 1_000_000_000,
            "milliards": 1_000_000_000,
            "billion": 1_000_000_000_000,
            "billions": 1_000_000_000_000,
            "trillion": 1_000_000_000_000_000_000,
            "trillions": 1_000_000_000_000_000_000,
        }
        self.multipliers_ordinal = {
            "centième": (100, "ème"),
            "millième": (1_000, "ème"),
            "millionième": (1_000_000, "ème"),
            "milliardième": (1_000_000_000, "ème"),
        }
        self.multipliers_suffixed = {**self.multipliers_ordinal}

        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "moins": "-",
            "plus": "+",
        }
        self.following_prefixers = {
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "livre": "£",
            "livres": "£",
            "centime": "¢",
            "centimes": "¢",
        }
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {
            "pour": {"cent": "%"},
            "pourcent": "%",
        }
        self.specials = {"et", "virgule"}

        self.words = set(
            [
                key
                for mapping in [
                    self.zeros,
                    self.ones,
                    self.ones_suffixed,
                    self.tens,
                    self.tens_suffixed,
                    self.multipliers,
                    self.multipliers_suffixed,
                    self.preceding_prefixers,
                    self.following_prefixers,
                    self.suffixers,
                    self.specials,
                ]
                for key in mapping
            ]
        )
        self.literal_words = {"un", "une"}

    def process_words(self, words: list[str]) -> Iterator[str]:
        prefix: str | None = None
        value: str | int | None = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: str | int):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for prev, current, next in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f:
                    if value is not None:
                        if isinstance(value, str) and value.endswith("."):
                            # concatenate decimals / ip address components
                            value = str(value) + str(current)
                            continue
                        else:
                            yield output(value)

                    prefix = current[0] if has_prefix else prefix
                    if f.denominator == 1:
                        value = f.numerator
                    else:
                        value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str):
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # 10 to 16
                    # In French, ones 10-16 can be added after soixante (60) for 70-76,
                    # or after quatre-vingt (80) for 90-96, as well as after multiples of 100
                    if value % 100 in (0, 60, 80):
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str):
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # 10 to 16
                    if value % 100 in (0, 60, 80):
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                # Special French vigesimal: "quatre vingt(s)" = 4 * 20 = 80
                if current in ("vingt", "vingts") and isinstance(value, int) and value % 100 in range(2, 10):
                    # Handle "quatre vingt" -> 80, also "deux cent quatre vingt" -> 280
                    base = (value // 100) * 100
                    digit = value % 100
                    value = base + digit * 20
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
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
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
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "et":
                    # "et" connects numbers in French: "vingt et un", "soixante et onze"
                    # ignore it between number words
                    if prev not in self.multipliers and prev not in self.tens:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "virgule":
                    # French decimal separator
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    raise ValueError(f"Unexpected token: {current}")
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # replace "et demi(e)" with "virgule cinq"
        results = []

        segments = re.split(r"\bet\s+demi(?:e)?\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("virgule cinq")
                else:
                    results.append("et demi")

        s = " ".join(results)

        # split hyphens between number words so the tokenizer can process them individually
        # e.g. "quatre-vingt-dix-sept" -> "quatre vingt dix sept"
        def split_number_hyphens(m):
            token = m.group(0)
            parts = token.split("-")
            if all(p.lower() in self.words or p.lower() in self.zeros for p in parts):
                return " ".join(parts)
            return token

        s = re.sub(r"\b[\w]+([-][\w]+)+\b", split_number_hyphens, s)

        # put a space at number/letter boundary
        s = re.sub(r"([a-zéèêëàâùûôîïç])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-zéèêëàâùûôîïç])", r"\1 \2", s)

        # remove spaces which could be a suffix (ordinal)
        s = re.sub(r"([0-9])\s+(er|ère|ème|nd|nde|e)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "€2 et ¢7" -> "€2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:et )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class FrenchTextNormalizer:
    def __init__(self):
        self.ignore_patterns = r"\b(euh|heu|hum|hmm|mm|mhm|mmm|uh|um|ah|oh|bah|ben)\b"
        self.replacers = {
            # common abbreviations
            r"\bm\.\b": "monsieur ",
            r"\bmme\b": "madame ",
            r"\bmlle\b": "mademoiselle ",
            r"\bdr\b": "docteur ",
            r"\bprof\b": "professeur ",
            r"\bst\b": "saint ",
            r"\bste\b": "sainte ",
            # common contractions / spoken forms
            r"\bqq\b": "quelques ",
            r"\btjrs\b": "toujours ",
            r"\bstp\b": "s'il te plaît ",
            r"\bsvp\b": "s'il vous plaît ",
            r"\bpq\b": "pourquoi ",
            r"\bbcp\b": "beaucoup ",
            r"\bpcq\b": "parce que ",
        }
        self.standardize_numbers = FrenchNumberNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        # French uses spaces or periods as thousand separators: "1.000" or "1 000" -> "1000"
        s = re.sub(r"(\d)\.(\d{3})(?!\d)", r"\1\2", s)
        s = re.sub(r"(\d) (\d{3})(?!\d)", r"\1\2", s)
        # French uses comma as decimal separator: "1,5" -> "1.5"
        s = re.sub(r"(\d),(\d)", r"\1.\2", s)

        s = re.sub(r"\.(?!\d)", " ", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep numeric symbols

        s = self.standardize_numbers(s)

        # Move numeric symbol after the number, e.g., "100 €" -> "€100"
        s = re.sub(r"(\d+)\s*([$¢€£])", r"\2\1", s)
        # Remove spaces between currency symbols and numbers
        s = re.sub(r"(?<=[$¢€£])\s+(?=\d)", "", s)

        # Remove prefix/suffix symbols that are not followed/preceded by numbers
        s = re.sub(r"[.$¢€£](?!\d)", " ", s)
        s = re.sub(r"(?<!\d)%", " ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        return s
