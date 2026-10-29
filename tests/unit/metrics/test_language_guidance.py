"""Tests for per-language judge guidance injection."""

import hashlib
import json
import os
import string
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from eva.metrics import language_guidance
from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.accuracy.speech_fidelity import SpeechFidelityMetric
from eva.metrics.base import BaseMetric
from eva.metrics.diagnostic.transcription_accuracy_key_entities import TranscriptionAccuracyKeyEntitiesMetric
from eva.metrics.diagnostic.tts_fidelity import TTSFidelityMetric
from eva.metrics.experience.conciseness import ConcisenessJudgeMetric
from eva.metrics.language_guidance import (
    DISPLAY_NAMES_BY_CODE,
    GUIDANCE_DIR,
    get_language_guidance_text,
    render_language_guidance,
)
from eva.metrics.versioning import _CURRENT_PROMPT_HASH
from eva.models.config import PipelineType, base_language
from eva.utils.prompt_manager import get_prompt_manager

from .conftest import make_judge_metric, make_metric_context

# Scores the simulated user against its own instructions, so it has no guidance slot.
JUDGES_WITHOUT_GUIDANCE = {"user_behavioral_fidelity"}

GOLDEN_EN_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "judge_prompts_en.json"
REGEN_GOLDEN_ENV = "EVA_REGEN_JUDGE_PROMPT_GOLDEN"


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
    assert render_language_guidance("some_metric", "fr") == "\n### European French-specific guidance\nFrench note.\n"


@pytest.mark.parametrize(
    "content, error",
    [
        ("- one bullet\n- another bullet\n", "expected a mapping"),
        ("fr:\n  - one bullet\n", "must be a string"),
    ],
)
def test_malformed_guidance_file_raises(guidance_dir, content, error):
    (guidance_dir / "bad_metric.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        get_language_guidance_text("bad_metric", "fr")


def _judge_templates() -> list[tuple[str, str, str]]:
    judge = get_prompt_manager().prompts["judge"]
    return [
        (metric, key, template)
        for metric, prompts in judge.items()
        if isinstance(prompts, dict)
        for key, template in prompts.items()
        if isinstance(template, str)
    ]


def test_every_judge_template_has_guidance_slot():
    for metric, key, template in _judge_templates():
        if metric in JUDGES_WITHOUT_GUIDANCE:
            assert "{language_guidance}" not in template, f"judge.{metric}.{key} should not have a guidance slot"
        else:
            assert "{language_guidance}" in template, f"judge.{metric}.{key} lacks {{language_guidance}}"


def test_every_guidance_file_is_valid():
    judge = get_prompt_manager().prompts["judge"]
    known_languages = {code for code in DISPLAY_NAMES_BY_CODE if base_language(code) != "en"}
    for path in GUIDANCE_DIR.glob("*.yaml"):
        metric = path.stem
        assert metric in judge, f"{path.name} has no judge prompt"
        assert metric not in JUDGES_WITHOUT_GUIDANCE, f"{metric} prompt has no guidance slot"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert isinstance(data, dict), f"{path.name} must be a mapping of language code to text"
        for lang, text in data.items():
            assert lang in known_languages, f"{path.name}: {lang!r} is not a supported non-English language code"
            assert isinstance(text, str), f"{path.name}: guidance for {lang!r} must be a string"
            assert "{" not in text and "}" not in text, f"{path.name}: braces are not allowed in guidance text"


def test_get_judge_prompt_injects_guidance_and_changes_hash(tmp_path, monkeypatch):
    metric = make_judge_metric(ConcisenessJudgeMetric)
    (tmp_path / f"{metric.name}.yaml").write_text(yaml.safe_dump({"fr": "French note."}), encoding="utf-8")
    monkeypatch.setattr(language_guidance, "GUIDANCE_DIR", tmp_path)
    language_guidance._load_metric_guidance.cache_clear()
    try:
        english = metric.get_judge_prompt(language="en", conversation_turns="<turns>")
        english_hash = _CURRENT_PROMPT_HASH.get()
        french = metric.get_judge_prompt(language="fr", conversation_turns="<turns>")
        french_hash = _CURRENT_PROMPT_HASH.get()
    finally:
        language_guidance._load_metric_guidance.cache_clear()

    assert "### European French-specific guidance\nFrench note.\n" in french
    assert "specific guidance" not in english
    assert french_hash != english_hash


class _Captured(Exception):
    """Raised by the patched get_judge_prompt to stop compute() once the prompt is requested."""


def _speech_fidelity_patches(metric):
    return [
        patch.object(metric, "load_role_audio", return_value=MagicMock()),
        patch.object(metric, "encode_audio_segment", return_value=""),
    ]


@pytest.mark.parametrize(
    "metric_cls, extra_patches",
    [
        # ConversationTextJudgeMetric
        (FaithfulnessJudgeMetric, lambda m: [patch.object(m, "format_transcript", return_value="<transcript>")]),
        # PerTurnConversationJudgeMetric
        (ConcisenessJudgeMetric, lambda m: [patch.object(m, "format_transcript", return_value="<transcript>")]),
        # SpeechFidelityBaseMetric
        (
            TTSFidelityMetric,
            lambda m: [
                *_speech_fidelity_patches(m),
                patch.object(m, "_get_intended_turns", return_value={1: "Hello"}),
            ],
        ),
        # SpeechFidelityMetric
        (
            SpeechFidelityMetric,
            lambda m: [
                *_speech_fidelity_patches(m),
                patch.object(m, "_build_redacted_trace", return_value=[]),
                patch.object(m, "_get_assistant_turn_ids", return_value=[1]),
                patch.object(m, "_format_redacted_trace", return_value="<trace>"),
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_judge_metrics_forward_context_language(metric_cls, extra_patches):
    metric = make_judge_metric(metric_cls, mock_llm=True)
    context = make_metric_context(language="fr-CA", conversation_trace=[])
    spy = MagicMock(side_effect=_Captured)
    patches = [patch.object(metric, "get_judge_prompt", spy), *extra_patches(metric)]
    for p in patches:
        p.start()
    try:
        await metric.compute(context)
    finally:
        for p in patches:
            p.stop()

    spy.assert_called_once()
    assert spy.call_args.kwargs["language"] == "fr-CA"


@pytest.mark.parametrize("pipeline_type", [PipelineType.CASCADE, PipelineType.S2S])
def test_key_entities_forwards_context_language(pipeline_type):
    metric = make_judge_metric(TranscriptionAccuracyKeyEntitiesMetric)
    context = make_metric_context(
        language="ja",
        pipeline_type=pipeline_type,
        intended_user_turns={0: "ABC123"},
        transcribed_user_turns={0: "ABC123"},
    )
    spy = MagicMock(side_effect=_Captured)
    with (
        patch.object(metric, "get_judge_prompt", spy),
        patch.object(metric, "_build_audio_native_conversation", return_value=("<conversation>", [1])),
        pytest.raises(_Captured),
    ):
        metric._build_prompt_and_turns(context)

    assert spy.call_args.kwargs["language"] == "ja"


def _render_english_judge_prompts() -> dict[str, str]:
    """Render every judge template in English with placeholder values, keyed by ``metric.prompt_key``."""
    prompt_manager = get_prompt_manager()
    hashes = {}
    for metric, key, template in _judge_templates():
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
        variables = {name: f"<{name}>" for name in fields - {"language_guidance"}}
        stub = SimpleNamespace(name=metric, prompt_manager=prompt_manager)
        rendered = BaseMetric.get_judge_prompt(stub, key, language="en", **variables)
        hashes[f"{metric}.{key}"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    return hashes


def test_english_judge_prompts_match_golden():
    """English judge prompts are byte-identical to the golden renders.

    An intended change to an English prompt must update the golden file:
    ``EVA_REGEN_JUDGE_PROMPT_GOLDEN=1 uv run pytest tests/unit/metrics/test_language_guidance.py``.
    """
    current = _render_english_judge_prompts()
    if os.environ.get(REGEN_GOLDEN_ENV):
        GOLDEN_EN_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN_EN_PATH.read_text())
    assert current == golden, f"English judge prompts changed. If intended, regenerate with {REGEN_GOLDEN_ENV}=1."
