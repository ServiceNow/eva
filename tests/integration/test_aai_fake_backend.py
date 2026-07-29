"""End-to-end bridge test against an in-process fake aai host.

Exercises both WebSocket hops without any paid API or Node process:
a fake user simulator sends Twilio-framed mu-law audio to AAIAssistantServer,
which speaks host mode to a fake aai server that requests a tool and replies
with audio.
"""

import asyncio
import audioop
import base64
import json
import time
from unittest.mock import MagicMock

import pytest
import websockets

from eva.assistant.aai_server import AAIAssistantServer
from eva.assistant.audio_bridge import create_twilio_media_message

FAKE_AAI_PORT = 38999
BRIDGE_PORT = 38998


class FakeAaiHost:
    """Minimal aai host: completes the handshake, calls one tool, speaks, ends the turn."""

    def __init__(self):
        self.handshake: dict | None = None
        self.tool_results: list[dict] = []
        self.received_audio_bytes = 0
        self._server = None

    async def start(self) -> None:
        self._server = await websockets.serve(self._handle, "localhost", FAKE_AAI_PORT)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws) -> None:
        # 1. Handshake: first text frame carries the host config.
        raw = await ws.recv()
        self.handshake = json.loads(raw)
        await ws.send(json.dumps({"type": "config"}))

        # 2. Ask the client to run a tool.
        await ws.send(
            json.dumps(
                {
                    "type": "tool_call",
                    "toolCallId": "call_1",
                    "toolName": "probe_tool",
                    "args": {"query": "status"},
                }
            )
        )

        # 3. Wait for the relayed result, tolerating interleaved user audio.
        while True:
            frame = await ws.recv()
            if isinstance(frame, bytes | bytearray):
                self.received_audio_bytes += len(frame)
                continue
            message = json.loads(frame)
            if message.get("type") == "tool_result":
                self.tool_results.append(message)
                break

        # 4. Speak: transcript, 100 ms of 24 kHz PCM16, then end the turn.
        await ws.send(json.dumps({"type": "agent_transcript", "text": "Your status is green."}))
        await ws.send(b"\x00\x01" * 2400)
        await ws.send(json.dumps({"type": "reply_done"}))
        await ws.send(json.dumps({"type": "audio_done"}))

        # Keep the socket open so the bridge can finish draining.
        await asyncio.sleep(1.0)


def _make_server() -> AAIAssistantServer:
    """Build a server with the heavy collaborators stubbed."""
    srv = object.__new__(AAIAssistantServer)

    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {
        "model": "aai-host",
        "ws_url": f"ws://localhost:{FAKE_AAI_PORT}/websocket",
    }
    srv.agent = MagicMock()
    srv.agent.tools = []
    srv.conversation_id = "conv-test"
    srv.port = BRIDGE_PORT
    srv.initial_message = "Thank you for calling."
    srv.audit_log = MagicMock()
    srv.tool_handler = MagicMock()

    async def _fake_execute(name, args):
        return {"status": "ok", "echo": args}

    srv.execute_tool = _fake_execute
    srv._build_system_prompt = lambda: "You are a test agent."
    srv.get_initial_scenario_db = lambda: {}
    srv.get_final_scenario_db = lambda: {}

    # Attributes normally set by the two __init__ methods.
    srv._model = "aai-host"
    srv._ws_url = f"ws://localhost:{FAKE_AAI_PORT}/websocket"
    srv._input_rate = 16000
    srv._output_rate = 24000
    srv._session = None
    srv._to_backend = None
    srv._from_backend = None
    srv._stream_sid = "conv-test"
    srv._audio_out = asyncio.Queue()
    srv._tasks = []
    srv._user_speech_start_ms = None
    srv._user_speech_stop_ms = None
    srv._user_speaking = False
    srv._assistant_speaking = False
    srv._turn_first_audio_ms = None
    srv._assistant_text = ""
    srv.user_audio_buffer = bytearray()
    srv.assistant_audio_buffer = bytearray()
    srv._audio_buffer = bytearray()
    srv._audio_sample_rate = 24000
    srv._app = None
    srv._server = None
    srv._server_task = None
    srv._running = False
    return srv


@pytest.fixture
async def fake_host():
    host = FakeAaiHost()
    await host.start()
    yield host
    await host.stop()


async def test_full_session_round_trip(fake_host, tmp_path):
    srv = _make_server()
    srv.output_dir = tmp_path
    await srv.start()

    received_audio = bytearray()
    try:
        async with websockets.connect(f"ws://localhost:{BRIDGE_PORT}/ws") as sim:
            await sim.send(json.dumps({"event": "start", "start": {"streamSid": "stream-1"}}))

            # 200 ms of silence as mu-law 8 kHz, in 20 ms Twilio frames.
            silence_mulaw = audioop.lin2ulaw(b"\x00\x00" * 1600, 2)
            for offset in range(0, len(silence_mulaw), 160):
                await sim.send(create_twilio_media_message("stream-1", silence_mulaw[offset : offset + 160]))
                await asyncio.sleep(0.005)

            await sim.send(json.dumps({"event": "user_speech_stop", "timestamp_ms": str(_now_ms())}))

            # Collect whatever the bridge paces back.
            deadline = asyncio.get_running_loop().time() + 3.0
            while asyncio.get_running_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(sim.recv(), timeout=0.3)
                except TimeoutError:
                    if received_audio:
                        break
                    continue
                message = json.loads(raw)
                if message.get("event") == "media":
                    received_audio.extend(base64.b64decode(message["media"]["payload"]))
    finally:
        await srv.stop()

    # Handshake carried the injected agent definition.
    assert fake_host.handshake is not None
    assert fake_host.handshake["type"] == "config"
    assert fake_host.handshake["audioFormat"] == "pcm16"
    assert fake_host.handshake["sampleRate"] == 16000
    assert fake_host.handshake["ttsSampleRate"] == 24000
    assert fake_host.handshake["host"]["systemPrompt"] == "You are a test agent."
    assert fake_host.handshake["host"]["greeting"] == "Thank you for calling."

    # User audio reached the backend.
    assert fake_host.received_audio_bytes > 0

    # The tool round-tripped through EVA, not the backend's sandbox.
    assert len(fake_host.tool_results) == 1
    assert fake_host.tool_results[0]["toolCallId"] == "call_1"
    assert json.loads(fake_host.tool_results[0]["result"])["status"] == "ok"

    # Assistant audio came back as 160-byte mu-law frames.
    assert len(received_audio) > 0
    assert len(received_audio) % 160 == 0

    # The turn was recorded.
    srv.audit_log.append_assistant_output.assert_called()
    assert srv.audit_log.append_assistant_output.call_args[0][0] == "Your status is green."


def _now_ms() -> int:
    return int(time.time() * 1000)
