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
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pipecat.transcriptions.language import Language

from eva.models.config import LANGUAGE_DISPLAY_NAMES

_CONFIGS_AGENTS = Path(__file__).resolve().parents[3] / "configs" / "agents"
INITIAL_MESSAGES_PATH = _CONFIGS_AGENTS / "initial_messages.yaml"

_LANGUAGE_ADDENDUM_TEMPLATE = (
    "Always respond to the user in {display_name}, regardless of the instructions"
    " given or tool outputs received. However, tool calls and tool names must always be"
    " done using ascii characters, except parameters like people's first or last names"
    " which may be in non-ascii, native script. You may need to try both scripts when"
    " looking up by name. All translatable values should be translated when talking to"
    " the user. For example, if you are telling the user about a location from a tool"
    " response which says 'Downtown' or 'West Laboratory', this should be translated. Only"
    " distinct item names (e.g. 'IntelliJ') should be kept in their original form."
)

FIRST_NAME_PLACEHOLDER = "<FIRST_NAME>"
LAST_NAME_PLACEHOLDER = "<LAST_NAME>"
FIRST_NAME_ROMANIZED_PLACEHOLDER = "<FIRST_NAME_ROMANIZED>"
LAST_NAME_ROMANIZED_PLACEHOLDER = "<LAST_NAME_ROMANIZED>"
PHONE_PLACEHOLDER = "<PHONE>"
# Companion = a named third party referenced in the scenario (e.g. the user's
# husband). Sampled per-language alongside the user's name (see add_culture_data.py),
# guaranteed distinct from the user's first name within the record.
COMPANION_FIRST_NAME_PLACEHOLDER = "<COMPANION_FIRST_NAME>"
COMPANION_FIRST_NAME_ROMANIZED_PLACEHOLDER = "<COMPANION_FIRST_NAME_ROMANIZED>"


def _replace_in(
    obj: Any,
    first: str,
    last: str,
    first_rom: str,
    last_rom: str,
    phone: str = "",
    companion_first: str = "",
    companion_first_rom: str = "",
) -> Any:
    if isinstance(obj, str):
        # Romanized placeholders are emitted lowercase: they exist for email
        # local-parts and similar ASCII slug contexts.
        result = (
            obj.replace(FIRST_NAME_ROMANIZED_PLACEHOLDER, first_rom.lower())
            .replace(LAST_NAME_ROMANIZED_PLACEHOLDER, last_rom.lower())
            .replace(FIRST_NAME_PLACEHOLDER, first)
            .replace(LAST_NAME_PLACEHOLDER, last)
        )
        if phone:
            result = result.replace(PHONE_PLACEHOLDER, phone)
        if companion_first:
            result = result.replace(COMPANION_FIRST_NAME_ROMANIZED_PLACEHOLDER, companion_first_rom.lower()).replace(
                COMPANION_FIRST_NAME_PLACEHOLDER, companion_first
            )
        return result
    if isinstance(obj, list):
        return [
            _replace_in(x, first, last, first_rom, last_rom, phone, companion_first, companion_first_rom) for x in obj
        ]
    if isinstance(obj, dict):
        return {
            k: _replace_in(v, first, last, first_rom, last_rom, phone, companion_first, companion_first_rom)
            for k, v in obj.items()
        }
    return obj


def _phone_for(culture_overrides: dict | None, language: str) -> str:
    """Extract phone number for ``language`` from ``culture_overrides``, or empty string."""
    if not culture_overrides or language not in culture_overrides:
        return ""
    return culture_overrides[language].get("phone", "")


def _companion_for(
    culture_overrides: dict | None,
    romanized_culture_overrides: dict | None,
    language: str,
) -> tuple[str, str]:
    """Return ``(companion_first, companion_first_rom)`` for ``language``, or empty strings.

    Records without a companion entry get empty strings, in which case the placeholder
    substitution is skipped (the placeholder stays literal and any reference to it would
    surface as a visible bug at runtime — desired loud-fail behavior).
    """
    if not culture_overrides or language not in culture_overrides:
        return "", ""
    comp = culture_overrides[language].get("companion") or {}
    first = comp.get("first_name", "")
    if not first:
        return "", ""
    rom_comp = (romanized_culture_overrides or {}).get(language, {}).get("companion") or {}
    first_rom = rom_comp.get("first_name") or first
    return first, first_rom


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
    phone = _phone_for(culture_overrides, language)
    comp_first, comp_first_rom = _companion_for(culture_overrides, romanized_culture_overrides, language)
    resolved = _replace_in(
        copy.deepcopy(user_goal), first, last, first_rom, last_rom, phone, comp_first, comp_first_rom
    )

    if not starting_utterances or language not in starting_utterances:
        raise KeyError(
            f"starting_utterances missing entry for language {language!r}. "
            f"Available: {list(starting_utterances or [])}. "
            f"Run scripts/add_culture_data.py --language {language} to populate it."
        )
    utt = starting_utterances[language]
    resolved["starting_utterance"] = _replace_in(
        utt, first, last, first_rom, last_rom, phone, comp_first, comp_first_rom
    )
    return resolved


def resolve_user_config(
    user_config: dict,
    culture_overrides: dict | None,
    language: str,
    romanized_culture_overrides: dict | None = None,
) -> dict:
    first, last, first_rom, last_rom = _names_for(culture_overrides, romanized_culture_overrides, language)
    phone = _phone_for(culture_overrides, language)
    comp_first, comp_first_rom = _companion_for(culture_overrides, romanized_culture_overrides, language)
    return _replace_in(copy.deepcopy(user_config), first, last, first_rom, last_rom, phone, comp_first, comp_first_rom)


def add_user_language_directive(language: str, language_display_name: str, user_persona: str) -> str | None:
    """Return the language directive appended to the user-simulator persona.

    Returns None for English (no directive needed). The same string is used both
    at runtime (injected into the simulator persona) and by the judge metric
    (so the judge sees the exact instruction the simulator received).
    """
    if not language or language.lower() in {"en", "english"}:
        return user_persona
    directive = (
        f"Speak ONLY in {language_display_name}. Do not switch to English even if the agent does. "
        "All translatable values should be translated when talking to the agent. "
        "For example, if you are telling the agent about a location like 'Downtown Office' and you are speaking Spanish, say 'oficina del centro'. "
        "If you are talking about a date that you read as MM/DD/YYYY, you should say it in the culturally appropriate format. "
        "Distinct proper names (e.g. 'IntelliJ', 'Google') should be kept in their original form."
    )
    return f"{user_persona}\n\n{directive}"


def get_language_addendum(language: str) -> str | None:
    """Return the agent prompt addendum for the language, or None for English/unknown."""
    if not language or language.lower() in ("en", "english"):
        return None
    return _LANGUAGE_ADDENDUM_TEMPLATE.format(display_name=LANGUAGE_DISPLAY_NAMES.get(Language(language)))


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


@lru_cache(maxsize=8)
def _load_aliases_index(aliases_dir: str) -> dict[str, dict]:
    """Load every ``<slug>.json`` under ``aliases_dir`` into a ``{canonical_name: payload}`` map.

    Cached per directory so repeated record-level calls don't reread the same files.
    Returns ``{}`` if the directory is missing — domains without an aliases store
    just skip injection.
    """
    p = Path(aliases_dir)
    if not p.exists():
        return {}
    index: dict[str, dict] = {}
    for f in sorted(p.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        index[data["name"]] = data
    return index


def _inject_aliases(obj: Any, index: dict[str, dict], language: str) -> None:
    """Walk ``obj`` in place, populating ``name_aliases`` on dicts whose ``name`` is in ``index``.

    ``name_aliases`` = base + ``translations[language]`` (deduped, order-preserved).
    Other dicts/lists are recursed into. Entries whose ``name`` is unknown are left alone.
    """
    if isinstance(obj, dict):
        name = obj.get("name")
        if isinstance(name, str) and name in index:
            entry = index[name]
            seen: set[str] = set()
            merged: list[str] = []
            for a in entry.get("base") + list((entry.get("translations")).get(language, [])):
                if a not in seen:
                    seen.add(a)
                    merged.append(a)
            obj["name_aliases"] = merged
        for v in obj.values():
            _inject_aliases(v, index, language)
    elif isinstance(obj, list):
        for item in obj:
            _inject_aliases(item, index, language)


def resolve_scenario_db(
    db: Any,
    culture_overrides: dict | None,
    language: str,
    romanized_culture_overrides: dict | None = None,
    aliases_dir: Path | str | None = None,
) -> Any:
    first, last, first_rom, last_rom = _names_for(culture_overrides, romanized_culture_overrides, language)
    phone = _phone_for(culture_overrides, language)
    comp_first, comp_first_rom = _companion_for(culture_overrides, romanized_culture_overrides, language)
    resolved = _replace_in(copy.deepcopy(db), first, last, first_rom, last_rom, phone, comp_first, comp_first_rom)
    if aliases_dir is not None:
        index = _load_aliases_index(str(aliases_dir))
        if index:
            _inject_aliases(resolved, index, language)
    return resolved
