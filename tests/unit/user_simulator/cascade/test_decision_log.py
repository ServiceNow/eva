"""Tests for the caller's tick trace."""

import json

from eva.user_simulator.cascade.decision_log import DecisionLog


def test_rows_are_written_one_json_object_per_line(tmp_path):
    log = DecisionLog(tmp_path / "trace.jsonl")
    log.log("tick", tick=1, has_assistant_speech=True)
    log.log("session_ended", tick=10, reason="goodbye")

    log.save()

    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["tick", "session_ended"]
    assert rows[0]["has_assistant_speech"] is True


def test_no_file_is_written_when_nothing_was_traced(tmp_path):
    DecisionLog(tmp_path / "trace.jsonl").save()

    assert not (tmp_path / "trace.jsonl").exists()


def test_rows_are_readable_before_save_so_a_crashed_run_still_has_a_trace(tmp_path):
    log = DecisionLog(tmp_path / "trace.jsonl")
    log.log("tick", tick=1)

    # No save() call: this is the killed-mid-run case.
    assert json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])["tick"] == 1


def test_save_is_idempotent(tmp_path):
    log = DecisionLog(tmp_path / "trace.jsonl")
    log.log("tick", tick=1)

    log.save()
    log.save()


def test_summary_counts_rows_by_kind(tmp_path):
    log = DecisionLog(tmp_path / "trace.jsonl")
    for _ in range(3):
        log.log("tick")
    log.log("session_ended")

    assert log.summary() == {"tick": 3, "session_ended": 1}
