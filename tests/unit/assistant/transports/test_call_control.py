"""Unit tests for the Telnyx Call Control transport."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web
import pytest

from eva.assistant.transports.call_control import CallControlTransport, _active_transports


class _FakeWebSocket:
    """Minimal stand-in for a FastAPI WebSocket."""

    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self._sent: list[str] = []
        self._closed = False
        self._idx = 0

    async def receive_text(self) -> str:
        if self._idx >= len(self._messages):
            # Simulate connection close
            raise Exception("WebSocket closed")
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_text(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        self._closed = True


class TestCallControlTransport:
    @pytest.mark.asyncio
    async def test_start_places_call_and_registers(self, unused_tcp_port: int) -> None:
        """start() places a call and registers in the global transport registry."""
        requests: list[dict] = []
        create_call_received = asyncio.Event()
        api_runner = await _start_api_server(unused_tcp_port, requests, create_call_received)

        transport = CallControlTransport(
            api_key="telnyx-key",
            to="sip:test@example.com",
            app_id="app-123",
            from_number="+15551234567",
            conversation_id="conv-test-1",
            webhook_base_url="https://example.ngrok-free.dev",
            api_base_url=f"http://127.0.0.1:{unused_tcp_port}/v2",
        )

        try:
            # start() will block waiting for media stream — run it in background
            # and simulate the stream connecting
            start_task = asyncio.create_task(transport.start())
            await asyncio.wait_for(create_call_received.wait(), timeout=5.0)

            # Verify the API call
            assert requests[0]["path"] == "/v2/calls"
            call_payload = requests[0]["json"]
            assert call_payload["connection_id"] == "app-123"
            assert call_payload["to"] == "sip:test@example.com"
            assert call_payload["from"] == "+15551234567"
            assert call_payload["stream_url"] == "wss://example.ngrok-free.dev/media-stream/conv-test-1"
            assert call_payload["stream_bidirectional_mode"] == "rtp"
            assert call_payload["stream_bidirectional_codec"] == "L16"
            assert call_payload["stream_bidirectional_sampling_rate"] == 16000

            # Transport should be registered globally
            assert _active_transports.get("conv-test-1") is transport

            # Simulate the media stream connecting (what the webhook WS handler would do)
            transport._connected_event.set()
            await asyncio.wait_for(start_task, timeout=2.0)

            assert transport._call_control_id == "call-control-123"
            assert transport._call_session_id == "session-abc-123"
            assert transport._eva_call_id is not None  # Generated before dialing

            # Verify the custom header was sent with our eva_call_id
            custom_headers = call_payload.get("custom_headers", [])
            eva_headers = [h for h in custom_headers if h["name"] == "X-Eva-Call-Id"]
            assert len(eva_headers) == 1
            assert eva_headers[0]["value"] == transport._eva_call_id
        finally:
            await transport.stop()
            await api_runner.cleanup()

        # After stop, should be unregistered
        assert "conv-test-1" not in _active_transports

    @pytest.mark.asyncio
    async def test_send_and_receive_audio(self) -> None:
        """Audio flows bidirectionally through the media stream WebSocket."""
        transport = CallControlTransport(
            api_key="telnyx-key",
            to="sip:test@example.com",
            app_id="app-123",
            from_number="+15551234567",
            conversation_id="conv-audio-1",
            webhook_base_url="https://example.ngrok-free.dev",
        )

        received_audio: list[bytes] = []
        transport.set_audio_handler(lambda audio: _append_audio(received_audio, audio))

        # Simulate inbound audio from Telnyx
        inbound_audio = b"\xff" * 160
        ws = _FakeWebSocket([
            json.dumps({"event": "connected"}),
            json.dumps({"event": "start", "stream_id": "stream-123"}),
            json.dumps({
                "event": "media",
                "media": {
                    "track": "inbound",
                    "payload": base64.b64encode(inbound_audio).decode("ascii"),
                },
            }),
            json.dumps({"event": "stop"}),
        ])

        await transport.handle_media_stream(ws)

        # Should have received the inbound audio
        assert received_audio == [inbound_audio]

        # Now test sending outbound audio
        transport._stream_ws = ws  # Re-attach for send test
        outbound_audio = b"\x7f" * 160
        await transport.send_audio(outbound_audio)

        assert len(ws._sent) == 1
        sent_msg = json.loads(ws._sent[0])
        assert sent_msg["event"] == "media"
        assert base64.b64decode(sent_msg["media"]["payload"]) == outbound_audio

    @pytest.mark.asyncio
    async def test_start_raises_when_call_creation_fails(self, unused_tcp_port: int) -> None:
        async def failing_create_call(_request: web.Request) -> web.Response:
            return web.json_response({"errors": [{"detail": "boom"}]}, status=500)

        app = web.Application()
        app.router.add_post("/v2/calls", failing_create_call)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
        await site.start()

        transport = CallControlTransport(
            api_key="telnyx-key",
            to="sip:test@example.com",
            app_id="app-123",
            from_number="+15551234567",
            conversation_id="conv-fail-1",
            webhook_base_url="https://example.ngrok-free.dev",
            api_base_url=f"http://127.0.0.1:{unused_tcp_port}/v2",
        )

        try:
            with pytest.raises(RuntimeError, match="500"):
                await transport.start()

            # After failed start + cleanup, session should be closed
            assert transport._session is None
            assert "conv-fail-1" not in _active_transports
        finally:
            await transport.stop()
            await runner.cleanup()


async def _start_api_server(port: int, requests: list[dict], create_call_received: asyncio.Event) -> web.AppRunner:
    async def create_call(request: web.Request) -> web.Response:
        requests.append({"path": request.path, "json": await request.json()})
        create_call_received.set()
        return web.json_response({"data": {"call_control_id": "call-control-123", "call_session_id": "session-abc-123"}})

    async def hangup(request: web.Request) -> web.Response:
        requests.append({"path": request.path, "json": await request.json()})
        return web.json_response({"data": {"result": "ok"}})

    app = web.Application()
    app.router.add_post("/v2/calls", create_call)
    app.router.add_post("/v2/calls/{call_control_id}/actions/hangup", hangup)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


async def _append_audio(received_audio: list[bytes], audio: bytes) -> None:
    received_audio.append(audio)
