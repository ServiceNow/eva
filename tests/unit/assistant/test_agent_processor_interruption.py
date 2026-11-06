"""Tests for re-running a query that an interruption cancelled when the interrupting turn is empty.

With the 'external' turn-start strategy a turn opens on the STT's turn.start, before any words.
If that turn.start came from trailing audio, the interruption cancels the in-flight query and the
turn then closes with no text. Without recovery nothing answers the user and the agent stalls.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.pipeline.agent_processor import BenchmarkAgentProcessor


@pytest.fixture
def proc(monkeypatch):
    """A bare BenchmarkAgentProcessor whose query processing is recorded instead of run."""
    monkeypatch.setattr("pipecat.processors.frame_processor.FrameProcessor._start_interruption", AsyncMock())

    p = object.__new__(BenchmarkAgentProcessor)
    p._pending_fallback = None
    p._current_query_task = None
    p._current_query_text = None
    p._cancelled_query_text = None
    p._interrupted = asyncio.Event()
    p._user_message_aggregator = []
    p._pending_final_transcripts = []
    p._latest_interim_transcript = ""
    p.fallback_timer = None
    p.audit_log = AuditLog()

    p.calls = []  # (text, log_user_input) for each query that ran to completion
    p.block = asyncio.Event()  # never set: a blocked query only ends by being cancelled
    p.started = asyncio.Event()  # set once a blocking query is in flight
    p.block_next = False  # when True, the next query blocks (to be interrupted)

    async def fake_process_user_query(text, log_user_input=True):
        if p.block_next:
            p.block_next = False
            p.started.set()
            await p.block.wait()
        p.calls.append((text, log_user_input))

    p._process_user_query = fake_process_user_query
    return p


async def _interrupt_in_flight_query(proc, text):
    """Start a turn whose query blocks, interrupt it, and wait for the turn handler to return."""
    proc.block_next = True
    proc.started.clear()
    turn = asyncio.create_task(proc.process_complete_user_turn(text))
    await asyncio.wait_for(proc.started.wait(), 1)  # the query is in flight
    await proc._start_interruption()
    await asyncio.wait_for(turn, 1)


async def test_empty_turn_after_cancelled_query_reruns_it(proc):
    await _interrupt_in_flight_query(proc, "I want to cancel my flight")
    assert proc.calls == []  # the interrupted query never finished
    assert proc._cancelled_query_text == "I want to cancel my flight"

    await proc.process_complete_user_turn("")  # the interrupting turn closes with no words

    # Re-run without logging the user input again: it is already in the audit log.
    assert proc.calls == [("I want to cancel my flight", False)]
    assert proc._cancelled_query_text is None  # consumed
    assert not proc._interrupted.is_set()  # the re-run is allowed to speak
    assert proc._user_message_aggregator == [{"role": "user", "content": "I want to cancel my flight"}]


async def test_recovery_happens_once(proc):
    await _interrupt_in_flight_query(proc, "hello")
    await proc.process_complete_user_turn("")
    await proc.process_complete_user_turn("")  # a second empty turn has nothing to recover

    assert proc.calls == [("hello", False)]


async def test_empty_turn_without_cancelled_query_is_ignored(proc):
    await proc.process_complete_user_turn("")
    await proc.process_complete_user_turn("   ")

    assert proc.calls == []


async def test_interruption_with_no_query_in_flight_records_nothing(proc):
    await proc._start_interruption()

    assert proc._cancelled_query_text is None
    await proc.process_complete_user_turn("")
    assert proc.calls == []


async def test_real_turn_after_cancelled_query_replaces_it(proc):
    await _interrupt_in_flight_query(proc, "I want to cancel my flight")

    await proc.process_complete_user_turn("Actually, change it instead")  # a real barge-in

    assert proc.calls == [("Actually, change it instead", True)]
    assert proc._cancelled_query_text is None
    await proc.process_complete_user_turn("")  # the old query is not resurrected later
    assert proc.calls == [("Actually, change it instead", True)]


async def test_recovered_query_can_be_interrupted_and_recovered_again(proc):
    await _interrupt_in_flight_query(proc, "hello")
    proc.block_next = True
    proc.started.clear()
    turn = asyncio.create_task(proc.process_complete_user_turn(""))  # recovery starts and blocks
    await asyncio.wait_for(proc.started.wait(), 1)
    assert proc._current_query_text == "hello"
    await proc._start_interruption()  # a second phantom turn interrupts the re-run
    await asyncio.wait_for(turn, 1)
    assert proc.calls == []

    await proc.process_complete_user_turn("")

    assert proc.calls == [("hello", False)]


async def test_fallback_turn_does_not_recover_a_cancelled_query(proc):
    await _interrupt_in_flight_query(proc, "hello")
    proc._pending_fallback = (20, "")  # the turn-end backstop framed this turn as a fallback

    await proc.process_complete_user_turn("")

    assert len(proc.calls) == 1
    text, log_user_input = proc.calls[0]
    assert log_user_input is False
    assert "hello" != text  # the nudge ran, not the cancelled query
    assert proc._cancelled_query_text is None
