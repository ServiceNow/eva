import asyncio
import base64
import json

import pytest
from websockets.exceptions import ConnectionClosedOK

from eva.user_simulator.cascade.stt import MAX_RECONNECT_ATTEMPTS, ScribeStreamingSTT, TranscriptBuffer


def test_partial_updates_replace_the_in_flight_text():
    buffer = TranscriptBuffer()

    buffer.apply_partial("Let me check")
    buffer.apply_partial("Let me check that for")

    assert buffer.in_flight == "Let me check that for"
    assert buffer.committed == ""


def test_commit_appends_and_clears_the_partial():
    buffer = TranscriptBuffer()
    buffer.apply_partial("Let me check that")

    buffer.commit("Let me check that for you.")

    assert buffer.committed == "Let me check that for you."
    assert buffer.in_flight == ""


def test_successive_commits_accumulate_with_spaces():
    buffer = TranscriptBuffer()

    buffer.commit("First sentence.")
    buffer.commit("Second sentence.")

    assert buffer.committed == "First sentence. Second sentence."


def test_current_text_marks_the_incomplete_utterance():
    buffer = TranscriptBuffer()
    buffer.commit("I found your booking.")
    buffer.apply_partial("It leaves on Thurs")

    assert buffer.current_text() == "I found your booking. It leaves on Thurs [CURRENTLY SPEAKING, INCOMPLETE]"


def test_current_text_omits_the_marker_when_nothing_is_in_flight():
    buffer = TranscriptBuffer()
    buffer.commit("I found your booking.")

    assert buffer.current_text() == "I found your booking."


def test_take_committed_drains_the_buffer():
    buffer = TranscriptBuffer()
    buffer.commit("All done.")

    assert buffer.take_committed() == "All done."
    assert buffer.committed == ""


class SuspendingFakeWebSocket:
    """Fake websocket whose recv() genuinely suspends (awaits a future) before returning."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent: list[dict] = []

    async def recv(self):
        await asyncio.sleep(0.001)
        if not self._messages:
            await asyncio.Event().wait()
        item = self._messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        pass


async def _make_stt(messages) -> tuple[ScribeStreamingSTT, SuspendingFakeWebSocket]:
    stt = ScribeStreamingSTT({"api_key": "test-key"})
    fake = SuspendingFakeWebSocket(messages)
    stt._ws = fake
    stt._receive_task = asyncio.create_task(stt._receive_loop())
    return stt, fake


async def test_partial_transcript_lands_in_in_flight():
    stt, _ = await _make_stt(
        [
            {"message_type": "session_started", "session_id": "abc", "config": {}},
            {"message_type": "partial_transcript", "text": "It leaves on Thurs"},
        ]
    )
    await asyncio.sleep(0.05)
    assert stt.buffer.in_flight == "It leaves on Thurs"
    await stt.stop()


async def test_committed_transcript_lands_in_committed():
    stt, _ = await _make_stt(
        [
            {"message_type": "committed_transcript", "text": "It leaves on Thursday."},
        ]
    )
    await asyncio.sleep(0.05)
    assert stt.buffer.committed == "It leaves on Thursday."
    await stt.stop()


async def test_feed_with_commit_sends_commit_true():
    stt, fake = await _make_stt([])
    await stt.feed(b"\x00\x00", commit=True)

    assert fake.sent[-1]["commit"] is True
    assert base64.b64decode(fake.sent[-1]["audio_base_64"]) == b"\x00\x00"
    await stt.stop()


async def test_feed_without_commit_omits_commit_flag():
    stt, fake = await _make_stt([])
    await stt.feed(b"\x00\x00")

    assert "commit" not in fake.sent[-1]
    await stt.stop()


async def test_session_started_is_ignored_harmlessly():
    stt, _ = await _make_stt(
        [
            {"message_type": "session_started", "session_id": "abc", "config": {}},
        ]
    )
    await asyncio.sleep(0.05)
    assert stt.buffer.committed == ""
    assert stt.buffer.in_flight == ""
    assert stt._error is None
    await stt.stop()


async def test_recv_error_is_recorded_and_object_stays_usable():
    stt, fake = await _make_stt([RuntimeError("boom")])
    await asyncio.sleep(0.05)

    assert isinstance(stt._error, RuntimeError)
    await stt.feed(b"\x00\x00")
    assert fake.sent
    await stt.stop()


class ClosingFakeWebSocket(SuspendingFakeWebSocket):
    """Fake websocket whose send() fails, as a server-side close does."""

    async def send(self, raw: str) -> None:
        raise ConnectionClosedError()


class ConnectionClosedError(Exception):
    pass


async def test_feed_raises_and_marks_closed_when_the_socket_send_fails():
    stt = ScribeStreamingSTT({"api_key": "test-key"})
    fake = ClosingFakeWebSocket([])
    stt._ws = fake
    stt._receive_task = asyncio.create_task(stt._receive_loop())

    with pytest.raises(RuntimeError, match="Scribe session closed"):
        await stt.feed(b"\x00\x00")

    assert stt._closed is True
    await stt.stop()


async def test_feed_raises_immediately_once_closed_without_resending():
    stt, fake = await _make_stt([])
    stt._closed = True

    with pytest.raises(RuntimeError, match="Scribe session is closed"):
        await stt.feed(b"\x00\x00")

    assert fake.sent == []
    await stt.stop()


def _connect_stub(sockets):
    remaining = list(sockets)

    async def _connect():
        return remaining.pop(0)

    return _connect


async def test_reconnects_after_a_clean_close_and_keeps_transcribing():
    stt = ScribeStreamingSTT({"api_key": "test-key"})
    stt.buffer.commit("Heard before the close.")
    stt._ws = SuspendingFakeWebSocket([ConnectionClosedOK(None, None)])
    second_socket = SuspendingFakeWebSocket(
        [{"message_type": "committed_transcript", "text": "Heard after reconnecting."}]
    )
    stt._connect = _connect_stub([second_socket])
    stt._receive_task = asyncio.create_task(stt._receive_loop())

    await asyncio.sleep(0.05)

    assert stt.buffer.committed == "Heard before the close. Heard after reconnecting."
    assert stt._ws is second_socket
    assert stt._error is None
    assert stt._closed is False
    await stt.stop()


async def test_reconnect_drops_in_flight_partial_but_keeps_committed_text():
    stt = ScribeStreamingSTT({"api_key": "test-key"})
    stt.buffer.commit("Already committed.")
    stt.buffer.apply_partial("mid-utterance when the close happened")
    stt._ws = SuspendingFakeWebSocket([ConnectionClosedOK(None, None)])
    stt._connect = _connect_stub([SuspendingFakeWebSocket([])])
    stt._receive_task = asyncio.create_task(stt._receive_loop())

    await asyncio.sleep(0.05)

    assert stt.buffer.committed == "Already committed."
    assert stt.buffer.in_flight == ""
    await stt.stop()


async def test_persistent_clean_closes_exhaust_the_cap_and_surface_an_error():
    stt = ScribeStreamingSTT({"api_key": "test-key"})
    sockets = [SuspendingFakeWebSocket([ConnectionClosedOK(None, None)]) for _ in range(MAX_RECONNECT_ATTEMPTS + 1)]
    stt._ws = sockets[0]
    stt._connect = _connect_stub(sockets[1:])
    stt._receive_task = asyncio.create_task(stt._receive_loop())

    await asyncio.sleep(0.05)

    assert isinstance(stt._error, ConnectionClosedOK)
    assert stt._closed is True
    with pytest.raises(RuntimeError, match="Scribe session is closed"):
        await stt.feed(b"\x00\x00")
    await stt.stop()
