"""Tests for the caller's out-of-turn decision trace."""

import json

from eva.user_simulator.cascade.decision_log import DecisionLog


def test_rows_are_written_one_json_object_per_line(tmp_path):
    log = DecisionLog(tmp_path / "trace.jsonl")
    log.log("tick", tick=1, has_assistant_speech=True)
    log.log("listener_check", tick=10, should_interrupt=False)

    log.save()

    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["tick", "listener_check"]
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
    log.log("listener_check")

    assert log.summary() == {"tick": 3, "listener_check": 1}


async def test_a_declined_check_is_still_recorded_with_its_raw_reply():
    # The whole point: a NO must be distinguishable from a check that never ran.
    from eva.user_simulator.cascade.decisions import ListenerDecisions
    from tests.unit.user_simulator.cascade.test_decisions import FakeLLM

    decisions = ListenerDecisions(
        FakeLLM(["NO", "NO"]), interrupt_prompt="{conversation_history}", backchannel_prompt="{conversation_history}"
    )

    verdict = await decisions.evaluate("agent talking", allow_interrupt=True, allow_backchannel=True)

    assert verdict.should_interrupt is False
    assert verdict.interrupt_trace.ran is True
    assert verdict.interrupt_trace.raw == "NO"


async def test_a_check_that_was_not_allowed_reports_that_it_never_ran():
    from eva.user_simulator.cascade.decisions import ListenerDecisions
    from tests.unit.user_simulator.cascade.test_decisions import FakeLLM

    decisions = ListenerDecisions(
        FakeLLM([]), interrupt_prompt="{conversation_history}", backchannel_prompt="{conversation_history}"
    )

    verdict = await decisions.evaluate("agent talking", allow_interrupt=False, allow_backchannel=False)

    assert verdict.interrupt_trace.ran is False
    assert verdict.interrupt_trace.raw == ""
