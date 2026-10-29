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
from pipecat.transcriptions.language import Language

from eva.models.config import LANGUAGE_DISPLAY_NAMES

GUIDANCE_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "prompts" / "judge_language_guidance"


@cache
def _load_metric_guidance(metric_name: str) -> dict[str, str]:
    path = GUIDANCE_DIR / f"{metric_name}.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(lang): text.strip() for lang, text in data.items() if isinstance(text, str) and text.strip()}


def get_language_guidance_text(metric_name: str, language: str) -> str:
    """Return the raw guidance text for a metric and language (empty if none)."""
    guidance = _load_metric_guidance(metric_name)
    base = language.split("-")[0]
    parts = [guidance[base]] if base in guidance else []
    if language != base and language in guidance:
        parts.append(guidance[language])
    return "\n\n".join(parts)


def render_language_guidance(metric_name: str, language: str) -> str:
    """Return the block that replaces ``{language_guidance}`` in a judge prompt (empty if none)."""
    text = get_language_guidance_text(metric_name, language)
    if not text:
        return ""
    try:
        display_name = LANGUAGE_DISPLAY_NAMES.get(Language(language), language)
    except ValueError:
        display_name = language
    return f"## {display_name}-specific guidance\n{text}\n"
