"""Parameterized tests for _extract_turns_from_history and _reconcile_transcript_with_tools.

Each test case is a minimal history (list of events) with expected outputs,
loaded from tests/fixtures/processor_histories.json.
"""

import json
from pathlib import Path

import pytest

from eva.metrics.processor import MetricsContextProcessor, _ProcessorContext
from eva.models.config import PipelineType

FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "processor_histories.json"

with open(FIXTURES_PATH) as f:
    TEST_CASES = json.load(f)

TEST_IDS = [case["id"] for case in TEST_CASES]


def _int_keys(d: dict) -> dict:
    """Convert string keys to int keys (JSON only supports string keys)."""
    return {int(k): v for k, v in d.items()}


def _convert_expected_value(key: str, value):
    """Convert expected values from JSON format to Python format.

    - Dict attributes (turns_*) need int keys.
    - Audio timestamp values are lists in JSON but tuples in Python.
    - Set attributes (*_interrupted_turns) are JSON lists → Python sets.
    """
    if isinstance(value, dict):
        converted = _int_keys(value)
        if "audio_timestamps" in key:
            converted = {k: [tuple(seg) for seg in v] if v is not None else None for k, v in converted.items()}
        return converted
    if key.endswith("_interrupted_turns"):
        return set(value)
    return value


@pytest.fixture(params=TEST_CASES, ids=TEST_IDS)
def case(request):
    return request.param


class TestExtractTurnsFromHistory:
    """Test _extract_turns_from_history with minimal synthetic histories."""

    def test_expected_outputs(self, case):
        ctx = _ProcessorContext()
        ctx.record_id = case["id"]
        ctx.history = case["history"]
        pipeline_type_str = case.get("pipeline_type", "cascade")
        ctx.pipeline_type = PipelineType(pipeline_type_str)

        MetricsContextProcessor._extract_turns_from_history(ctx)
        MetricsContextProcessor._reconcile_transcript_with_tools(ctx)

        for key, expected_value in case["expected"].items():
            actual = getattr(ctx, key)
            expected = _convert_expected_value(key, expected_value)
            if key == "conversation_trace":
                # Strip timestamps for comparison (exact ms values are brittle)
                actual = [{k: v for k, v in entry.items() if k != "timestamp"} for entry in actual]
            assert actual == expected, (
                f"Case '{case['id']}', attribute '{key}':\n  expected: {expected}\n  actual:   {actual}"
            )
