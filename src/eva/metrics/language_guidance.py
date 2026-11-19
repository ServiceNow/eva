"""Per-language guidance blocks injected into judge prompts.

Each judge prompt template contains a ``{language_guidance}`` placeholder. For English it
renders to an empty string, so the rendered English prompt is unchanged. For other languages
it renders the guidance written by native speakers in
``configs/prompts/judge_language_guidance/<metric_name>.yaml``, a mapping of language code to
guidance text. A regional variant (e.g. ``fr-CA``) gets its base language's guidance
(``fr``) followed by its own entry, if any.
"""

from functools import cache
from pathlib import Path

import yaml

from eva.models.config import LANGUAGE_DISPLAY_NAMES, base_language

GUIDANCE_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "prompts" / "judge_language_guidance"

DISPLAY_NAMES_BY_CODE = {str(lang.value): name for lang, name in LANGUAGE_DISPLAY_NAMES.items()}


@cache
def _load_metric_guidance(metric_name: str) -> dict[str, str]:
    path = GUIDANCE_DIR / f"{metric_name}.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of language code to guidance text, got {type(data).__name__}")
    guidance = {}
    for lang, text in data.items():
        if not isinstance(text, str):
            raise ValueError(f"{path}: guidance for {lang!r} must be a string, got {type(text).__name__}")
        if text.strip():
            guidance[str(lang)] = text.strip()
    return guidance


def get_language_guidance_text(metric_name: str, language: str) -> str:
    """Return the raw guidance text for a metric and language (empty if none)."""
    guidance = _load_metric_guidance(metric_name)
    base = base_language(language)
    parts = [guidance[base]] if base in guidance else []
    if language != base and language in guidance:
        parts.append(guidance[language])
    return "\n\n".join(parts)


def render_language_guidance(metric_name: str, language: str) -> str:
    """Return the block that replaces ``{language_guidance}`` in a judge prompt (empty if none).

    The block starts with a newline so it is separated from the section before it. English
    renders empty, which leaves the template's own blank line as it was.
    """
    text = get_language_guidance_text(metric_name, language)
    if not text:
        return ""
    display_name = DISPLAY_NAMES_BY_CODE.get(language, language)
    return f"\n### {display_name}-specific guidance\n{text}\n"


_GENDER_WORDS = {"M": "male", "F": "female"}


def render_assistant_gender_note(gender: str | None, language: str, language_display_name: str) -> str:
    """Faithfulness ``wrong_language`` item for the assistant's assigned speaking gender.

    Empty for English or when no gender was assigned (the default, and always for S2S), so those
    prompts are unchanged. Otherwise it starts with a newline and continues the "IS a flag" list.
    """
    word = _GENDER_WORDS.get(gender or "")
    if not word or base_language(language) == "en":
        return ""
    return (
        f"\n- Self-references in the wrong grammatical gender. The assistant was instructed to speak as a {word} "
        f"speaker: where {language_display_name} marks the speaker's gender (verb forms, participles, adjectives the "
        f"assistant uses about itself), it must use {word} forms. Rate this 2, never 1, even when it recurs. Forms "
        "that refer to the user or to other people are not covered."
    )


def render_user_gender_note(gender: str | None, language: str, language_display_name: str) -> str:
    """user_behavioral_fidelity Corruption Type 6 item for the user simulator's speaking gender.

    Empty for English or when the persona has no known gender, so those prompts are unchanged.
    Otherwise it starts with a newline and continues the Type 6 list.
    """
    word = _GENDER_WORDS.get(gender or "")
    if not word or base_language(language) == "en":
        return ""
    other = "female" if word == "male" else "male"
    return (
        f"\n- The user speaks with a {word} voice. Also flag this type if the user consistently referred to themselves "
        f"with {other} grammatical forms where {language_display_name} marks the speaker's gender (verb forms, "
        "participles, adjectives about themselves); report it under wrong_language. Isolated slips are acceptable, "
        "and forms that refer to the agent or to other people are not covered."
    )
