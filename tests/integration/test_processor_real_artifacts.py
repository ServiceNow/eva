"""Regression test for MetricsContextProcessor using real conversation artifacts.

Runs the full postprocessor pipeline (build_history → extract_turns → load_latencies
→ reconcile_transcript) against the 1.1.2 record artifacts and compares the resulting
_ProcessorContext against a golden snapshot in expected_processor_context.json.
"""

import json
from pathlib import Path

import pytest

from eva.metrics.processor import MetricsContextProcessor, _ProcessorContext
from eva.models.results import ConversationResult

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "records" / "1.1.2"
EXPECTED_PATH = ARTIFACTS_DIR / "expected_processor_context.json"

# Fields that use int keys in Python but string keys in JSON
INT_KEY_FIELDS = {
    "transcribed_assistant_turns",
    "transcribed_user_turns",
    "intended_assistant_turns",
    "intended_user_turns",
    "audio_timestamps_assistant_turns",
    "audio_timestamps_user_turns",
}

# Fields that are sets in Python but lists in JSON
SET_FIELDS = {"assistant_interrupted_turns", "user_interrupted_turns"}


def _convert_expected_value(key: str, value):
    """Convert JSON-serialized expected values back to Python types."""
    if key in INT_KEY_FIELDS and isinstance(value, dict):
        converted = {int(k): v for k, v in value.items()}
        if "audio_timestamps" in key:
            converted = {k: [tuple(seg) for seg in v] if v is not None else None for k, v in converted.items()}
        return converted
    if key in SET_FIELDS:
        return set(value)
    return value


@pytest.fixture(scope="module")
def processor_context() -> _ProcessorContext:
    """Run the full postprocessor on the 1.1.2 artifacts and return the context."""
    result_data = json.loads((ARTIFACTS_DIR / "result.json").read_text())
    # Fix paths to point to the local artifacts directory
    result_data["output_dir"] = str(ARTIFACTS_DIR)
    result_data["pipecat_logs_path"] = str(ARTIFACTS_DIR / "pipecat_logs.jsonl")
    result_data["elevenlabs_logs_path"] = str(ARTIFACTS_DIR / "elevenlabs_events.jsonl")
    result_data["audio_mixed_path"] = str(ARTIFACTS_DIR / "audio_mixed.wav")
    result_data["audio_assistant_path"] = None
    result_data["audio_user_path"] = None

    result = ConversationResult(**result_data)
    processor = MetricsContextProcessor()
    ctx = processor.process_record(result, ARTIFACTS_DIR)
    assert ctx is not None, "Postprocessor returned None — processing failed"
    return ctx


@pytest.fixture(scope="module")
def expected_context() -> dict:
    """Load the golden expected processor context."""
    return json.loads(EXPECTED_PATH.read_text())


class TestProcessorRealArtifacts:
    """Verify the full postprocessor pipeline against real 1.1.2 artifacts."""

    def test_turn_counts(self, processor_context, expected_context):
        assert processor_context.num_assistant_turns == expected_context["num_assistant_turns"]
        assert processor_context.num_user_turns == expected_context["num_user_turns"]
        assert processor_context.num_tool_calls == expected_context["num_tool_calls"]

    def test_tools_called(self, processor_context, expected_context):
        assert processor_context.tool_called == expected_context["tool_called"]

    def test_conversation_metadata(self, processor_context, expected_context):
        assert processor_context.conversation_ended_reason == expected_context["conversation_ended_reason"]
        assert processor_context.pipeline_type.value == expected_context.get("pipeline_type", "cascade")

    def test_transcribed_assistant_turns(self, processor_context, expected_context):
        expected = _convert_expected_value(
            "transcribed_assistant_turns", expected_context["transcribed_assistant_turns"]
        )
        assert processor_context.transcribed_assistant_turns == expected

    def test_transcribed_user_turns(self, processor_context, expected_context):
        expected = _convert_expected_value("transcribed_user_turns", expected_context["transcribed_user_turns"])
        assert processor_context.transcribed_user_turns == expected

    def test_intended_assistant_turns(self, processor_context, expected_context):
        expected = _convert_expected_value("intended_assistant_turns", expected_context["intended_assistant_turns"])
        assert processor_context.intended_assistant_turns == expected

    def test_intended_user_turns(self, processor_context, expected_context):
        expected = _convert_expected_value("intended_user_turns", expected_context["intended_user_turns"])
        assert processor_context.intended_user_turns == expected

    def test_audio_timestamps_assistant(self, processor_context, expected_context):
        expected = _convert_expected_value(
            "audio_timestamps_assistant_turns", expected_context["audio_timestamps_assistant_turns"]
        )
        assert processor_context.audio_timestamps_assistant_turns == expected

    def test_audio_timestamps_user(self, processor_context, expected_context):
        expected = _convert_expected_value(
            "audio_timestamps_user_turns", expected_context["audio_timestamps_user_turns"]
        )
        assert processor_context.audio_timestamps_user_turns == expected

    def test_tool_params(self, processor_context, expected_context):
        assert processor_context.tool_params == expected_context["tool_params"]

    def test_tool_responses(self, processor_context, expected_context):
        assert processor_context.tool_responses == expected_context["tool_responses"]

    def test_conversation_trace(self, processor_context, expected_context):
        # Strip timestamps — exact ms values are brittle
        actual = [
            {k: v for k, v in entry.items() if k != "timestamp"} for entry in processor_context.conversation_trace
        ]
        assert actual == expected_context["conversation_trace"]

    def test_interrupted_turns(self, processor_context, expected_context):
        assert processor_context.assistant_interrupted_turns == set(expected_context["assistant_interrupted_turns"])
        assert processor_context.user_interrupted_turns == set(expected_context["user_interrupted_turns"])

    def test_latency_assistant_turns(self, processor_context, expected_context):
        assert processor_context.latency_assistant_turns == {
            int(k): v for k, v in expected_context["latency_assistant_turns"].items()
        }
