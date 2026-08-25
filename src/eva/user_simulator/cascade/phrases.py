"""Per-language phrase vocabularies for the caller's out-of-turn behavior.

Data rather than constants because these are *words*, and words are language
specific. Timing thresholds stay in `constants.py` — varying those across runs
makes metrics incomparable, whereas speaking English continuers into a French
conversation is simply wrong.

Generated per language by `scripts/add_culture_data.py`, alongside the initial
message it already translates, so a run only ever reads this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eva.utils.logging import get_logger

logger = get_logger(__name__)

PHRASES_PATH = Path(__file__).resolve().parents[4] / "configs" / "caller_phrases.yaml"

FALLBACK_LANGUAGE = "en"

_cache: dict[str, CallerPhrases] = {}


@dataclass(frozen=True)
class CallerPhrases:
    """The fixed things the caller can say without taking a turn."""

    backchannels: list[str]
    """Continuers: "I'm listening, keep going"."""

    barge_in_openers: list[str]
    """Opening fragment of an interruption, voiced ahead of the content."""

    @property
    def vocabulary(self) -> list[str]:
        """Every phrase that needs pre-rendering, in a stable order."""
        return [*self.backchannels, *self.barge_in_openers]


def _read_file() -> dict[str, Any]:
    """Load the phrase file, or return empty when it is missing or unreadable."""
    try:
        data = yaml.safe_load(PHRASES_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning(f"Caller phrase file unreadable ({exc}); out-of-turn behavior has no vocabulary")
        return {}
    return data or {}


def candidate_languages(language: str) -> list[str]:
    """Language tags to try, most specific first: 'fr-CA' -> 'fr-CA', 'fr', 'en'.

    A regional variant nearly always shares its continuers with the base language,
    so falling back to it is far better than falling back to English.
    """
    candidates = [language]
    if "-" in language:
        candidates.append(language.split("-", 1)[0])
    if FALLBACK_LANGUAGE not in candidates:
        candidates.append(FALLBACK_LANGUAGE)
    return candidates


def load_phrases(language: str) -> CallerPhrases:
    """Return the caller's phrase vocabulary for a language.

    Falls back to the base language and then to English rather than raising: a
    missing translation should degrade the realism of the behavior, not abort the
    run. The fallback is logged because English continuers in a non-English call
    are a defect in the data, not an acceptable outcome.
    """
    if language in _cache:
        return _cache[language]

    data = _read_file()
    for candidate in candidate_languages(language):
        entry = data.get(candidate)
        if not entry:
            continue
        if candidate != language:
            logger.warning(
                f"No caller phrases for {language!r}; falling back to {candidate!r}. "
                f"Run scripts/add_culture_data.py --language {language} to generate them."
            )
        phrases = CallerPhrases(
            backchannels=list(entry.get("backchannels") or []),
            barge_in_openers=list(entry.get("barge_in_openers") or []),
        )
        _cache[language] = phrases
        return phrases

    logger.error(f"No caller phrases for {language!r} and no {FALLBACK_LANGUAGE!r} fallback in {PHRASES_PATH}")
    empty = CallerPhrases(backchannels=[], barge_in_openers=[])
    _cache[language] = empty
    return empty
