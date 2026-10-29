"""Tests for per-language judge guidance injection."""

import pytest
import yaml

from eva.metrics import language_guidance
from eva.metrics.language_guidance import GUIDANCE_DIR, get_language_guidance_text, render_language_guidance
from eva.utils.prompt_manager import get_prompt_manager


@pytest.fixture
def guidance_dir(tmp_path, monkeypatch):
    (tmp_path / "some_metric.yaml").write_text(
        yaml.safe_dump({"fr": "French note.", "fr-CA": "Quebec note.", "de": "  "}), encoding="utf-8"
    )
    monkeypatch.setattr(language_guidance, "GUIDANCE_DIR", tmp_path)
    language_guidance._load_metric_guidance.cache_clear()
    yield tmp_path
    language_guidance._load_metric_guidance.cache_clear()


def test_english_and_missing_metric_render_empty(guidance_dir):
    assert render_language_guidance("some_metric", "en") == ""
    assert render_language_guidance("unknown_metric", "fr") == ""
    assert render_language_guidance("some_metric", "de") == ""


def test_regional_variant_gets_base_then_own_guidance(guidance_dir):
    assert get_language_guidance_text("some_metric", "fr") == "French note."
    assert get_language_guidance_text("some_metric", "fr-CA") == "French note.\n\nQuebec note."


def test_rendered_block_has_header(guidance_dir):
    assert render_language_guidance("some_metric", "fr") == "## European French-specific guidance\nFrench note.\n"


def test_every_guidance_file_matches_a_judge_prompt_with_placeholder():
    judge = get_prompt_manager().prompts["judge"]
    for path in GUIDANCE_DIR.glob("*.yaml"):
        metric = path.stem
        assert metric in judge, f"{path.name} has no judge prompt"
        templates = [t for t in judge[metric].values() if isinstance(t, str)]
        assert all("{language_guidance}" in t for t in templates), f"{metric} prompt lacks {{language_guidance}}"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "en" not in data, f"{path.name} must not define English guidance"
        for text in data.values():
            assert "{" not in text and "}" not in text, f"{path.name}: braces are not allowed in guidance text"
