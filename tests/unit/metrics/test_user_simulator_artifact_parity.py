from pathlib import Path

from eva.metrics.processor import MetricsContextProcessor, _ProcessorContext
from eva.models.config import PipelineType
from eva.user_simulator.event_logger import UserSimulatorEventLogger


def _processed_context(path: Path) -> _ProcessorContext:
    context = _ProcessorContext()
    context.record_id = "artifact-parity"
    context.pipeline_type = PipelineType.S2S
    context.history = MetricsContextProcessor._load_user_simulator_logs(str(path))
    MetricsContextProcessor._extract_turns_from_history(context)
    MetricsContextProcessor._reconcile_transcript_with_tools(context)
    return context


def test_neutral_and_legacy_simulator_artifacts_produce_identical_metric_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("eva.user_simulator.event_logger.time.time", lambda: 100.0)
    neutral_path = tmp_path / "user_simulator_events.jsonl"
    legacy_path = tmp_path / "elevenlabs_events.jsonl"
    event_logger = UserSimulatorEventLogger(
        neutral_path,
        provider="elevenlabs",
        legacy_output_path=legacy_path,
    )

    event_logger.log_audio_start("assistant")
    event_logger.log_assistant_speech("Hello, how can I help?")
    event_logger.log_audio_end("assistant")
    event_logger.log_audio_start("simulated_user")
    event_logger.log_user_speech("Please reset my password.")
    event_logger.log_audio_end("simulated_user")
    event_logger.log_audio_start("assistant")
    event_logger.log_assistant_speech("Your password has been reset.")
    event_logger.log_audio_end("assistant")
    event_logger.log_connection_state("session_ended", {"reason": "goodbye"})
    event_logger.save()

    neutral = _processed_context(neutral_path)
    legacy = _processed_context(legacy_path)
    metric_inputs = (
        "transcribed_assistant_turns",
        "transcribed_user_turns",
        "intended_assistant_turns",
        "intended_user_turns",
        "audio_timestamps_assistant_turns",
        "audio_timestamps_user_turns",
        "assistant_interrupted_turns",
        "user_interrupted_turns",
        "conversation_trace",
        "num_assistant_turns",
        "num_user_turns",
    )

    for attribute in metric_inputs:
        assert getattr(neutral, attribute) == getattr(legacy, attribute), attribute
