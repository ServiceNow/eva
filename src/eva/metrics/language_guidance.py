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
