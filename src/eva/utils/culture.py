"""Culture/language placeholder resolution for evaluation records.

Datasets and scenario databases store person identity as placeholder tokens. The
actual per-language values live on the record under ``culture_overrides`` and
``romanized_culture_overrides``. Resolution happens just-in-time before data is
handed to the simulator, the assistant tool runtime, or metrics.

Placeholders:
- ``<FIRST_NAME>`` / ``<LAST_NAME>``           — substituted from ``culture_overrides[lang]``
                                                  (may contain non-ASCII script).
- ``<FIRST_NAME_ROMANIZED>`` / ``<LAST_NAME_ROMANIZED>``
                                                — substituted from
                                                  ``romanized_culture_overrides[lang]``
                                                  (ASCII; used for email local-parts etc).

Translated opening utterances live at ``user_goal.<language>_starting_utterance``.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIGS_AGENTS = Path(__file__).resolve().parents[3] / "configs" / "agents"
LANGUAGE_ADDENDA_PATH = _CONFIGS_AGENTS / "language_addenda.yaml"
INITIAL_MESSAGES_PATH = _CONFIGS_AGENTS / "initial_messages.yaml"

FIRST_NAME_PLACEHOLDER = "<FIRST_NAME>"
LAST_NAME_PLACEHOLDER = "<LAST_NAME>"
FIRST_NAME_ROMANIZED_PLACEHOLDER = "<FIRST_NAME_ROMANIZED>"
LAST_NAME_ROMANIZED_PLACEHOLDER = "<LAST_NAME_ROMANIZED>"


def _replace_in(obj: Any, first: str, last: str, first_rom: str, last_rom: str) -> Any:
    if isinstance(obj, str):
        # Romanized placeholders are emitted lowercase: they exist for email
        # local-parts and similar ASCII slug contexts.
        return (
            obj.replace(FIRST_NAME_ROMANIZED_PLACEHOLDER, first_rom.lower())
            .replace(LAST_NAME_ROMANIZED_PLACEHOLDER, last_rom.lower())
            .replace(FIRST_NAME_PLACEHOLDER, first)
            .replace(LAST_NAME_PLACEHOLDER, last)
        )
    if isinstance(obj, list):
        return [_replace_in(x, first, last, first_rom, last_rom) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_in(v, first, last, first_rom, last_rom) for k, v in obj.items()}
    return obj


def _names_for(
    culture_overrides: dict | None,
    romanized_culture_overrides: dict | None,
    language: str,
) -> tuple[str, str, str, str]:
    if not culture_overrides or language not in culture_overrides:
        raise KeyError(
            f"culture_overrides missing entry for language {language!r}. Available: {list(culture_overrides or [])}."
        )
    rom = romanized_culture_overrides or {}
    if language not in rom:
        # English (and any already-ASCII culture) can fall back to the same names.
        rom_entry = culture_overrides[language]
    else:
        rom_entry = rom[language]
    entry = culture_overrides[language]
    return entry["first_name"], entry["last_name"], rom_entry["first_name"], rom_entry["last_name"]


def resolve_user_goal(
    user_goal: dict,
    culture_overrides: dict | None,
    language: str,
    romanized_culture_overrides: dict | None = None,
    starting_utterances: dict | None = None,
) -> dict:
    """Return a deep copy of ``user_goal`` with placeholders resolved.

    The active ``starting_utterance`` is injected from ``starting_utterances[language]``.

    The simulator only sees the chosen language's utterance; the full per-language
    dict is kept off the goal payload to avoid leaking other-language context.
    """
    first, last, first_rom, last_rom = _names_for(culture_overrides, romanized_culture_overrides, language)
    resolved = _replace_in(copy.deepcopy(user_goal), first, last, first_rom, last_rom)

    if not starting_utterances or language not in starting_utterances:
        raise KeyError(
            f"starting_utterances missing entry for language {language!r}. "
            f"Available: {list(starting_utterances or [])}. "
            f"Run scripts/add_culture_data.py --language {language} to populate it."
        )
    utt = starting_utterances[language]
    resolved["starting_utterance"] = _replace_in(utt, first, last, first_rom, last_rom)
    return resolved


def resolve_user_config(
    user_config: dict,
    culture_overrides: dict | None,
    language: str,
    romanized_culture_overrides: dict | None = None,
) -> dict:
    first, last, first_rom, last_rom = _names_for(culture_overrides, romanized_culture_overrides, language)
    return _replace_in(copy.deepcopy(user_config), first, last, first_rom, last_rom)


@lru_cache(maxsize=1)
def _load_language_addenda() -> dict[str, str]:
    if not LANGUAGE_ADDENDA_PATH.exists():
        return {}
    return yaml.safe_load(LANGUAGE_ADDENDA_PATH.read_text(encoding="utf-8")) or {}


def get_user_language_directive(language: str, language_display_name: str) -> str | None:
    """Return the language directive appended to the user-simulator persona.

    Returns None for English (no directive needed). The same string is used both
    at runtime (injected into the simulator persona) and by the judge metric
    (so the judge sees the exact instruction the simulator received).
    """
    if not language or language.lower() in {"en", "english"}:
        return None
    return (
        f"Speak ONLY in {language_display_name}. Do not switch to English even if the agent does. "
        "All translatable values should be translated when talking to the agent. "
        "For example, if you are telling the agent about a location like 'downtown' or a date, "
        "this should be translated. Distinct proper names (e.g. 'IntelliJ') "
        "should be kept in their original form."
    )


def get_language_addendum(language: str) -> str | None:
    """Return the agent prompt addendum for the language, or None for English/unknown."""
    if not language or language.lower() in {"en", "english"}:
        return None
    return _load_language_addenda().get(language)


@lru_cache(maxsize=1)
def _load_initial_messages() -> dict[str, str]:
    if not INITIAL_MESSAGES_PATH.exists():
        return {}
    return yaml.safe_load(INITIAL_MESSAGES_PATH.read_text(encoding="utf-8")) or {}


def get_initial_message(language: str) -> str:
    """Return the assistant's opening line for ``language``.

    Falls back to English. Raises if even English is missing.
    """
    msgs = _load_initial_messages()
    if language in msgs:
        return msgs[language]
    if "en" in msgs:
        return msgs["en"]
    raise KeyError(f"initial_messages.yaml missing 'en' fallback (looked up {language!r})")


def resolve_scenario_db(
    db: Any,
    culture_overrides: dict | None,
    language: str,
    romanized_culture_overrides: dict | None = None,
) -> Any:
    first, last, first_rom, last_rom = _names_for(culture_overrides, romanized_culture_overrides, language)
    return _replace_in(copy.deepcopy(db), first, last, first_rom, last_rom)
