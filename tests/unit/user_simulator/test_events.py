import json

from eva.user_simulator.events import resolve_user_simulator_events_path


def test_resolver_prefers_artifact_in_current_run_directory(tmp_path):
    current_dir = tmp_path / "copied-run"
    original_dir = tmp_path / "original-run"
    current_dir.mkdir()
    original_dir.mkdir()
    current_path = current_dir / "user_simulator_events.jsonl"
    original_path = original_dir / "user_simulator_events.jsonl"
    current_path.write_text(json.dumps({"source": "current"}))
    original_path.write_text(json.dumps({"source": "original"}))

    resolved = resolve_user_simulator_events_path(current_dir, str(original_path))

    assert resolved == current_path


def test_resolver_prefers_current_legacy_artifact_over_original_neutral_file(tmp_path):
    current_dir = tmp_path / "copied-run"
    original_dir = tmp_path / "original-run"
    current_dir.mkdir()
    original_dir.mkdir()
    current_path = current_dir / "elevenlabs_events.jsonl"
    original_path = original_dir / "user_simulator_events.jsonl"
    current_path.write_text(json.dumps({"source": "current"}))
    original_path.write_text(json.dumps({"source": "original"}))

    resolved = resolve_user_simulator_events_path(current_dir, str(original_path))

    assert resolved == current_path
