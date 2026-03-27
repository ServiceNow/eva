"""Unit tests for the tool webhook service."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from eva.assistant.tool_webhook import ToolWebhookService


@pytest.fixture
async def webhook_service() -> ToolWebhookService:
    service = ToolWebhookService(port=9888)
    yield service
    await service.stop()


async def _make_client(service: ToolWebhookService) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=service.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestToolWebhookService:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, webhook_service: ToolWebhookService):
        async with await _make_client(webhook_service) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_executes_tool_with_direct_body(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success", "data": {"confirmation_number": "ABC123"}})
        await webhook_service.register_conversation("call-123", executor)

        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/call-123/get_reservation", json={"reservation_id": "ABC123"})

        assert response.status_code == 200
        assert response.json() == {"status": "success", "data": {"confirmation_number": "ABC123"}}
        executor.execute.assert_awaited_once_with("get_reservation", {"reservation_id": "ABC123"})

    @pytest.mark.asyncio
    async def test_executes_tool_with_function_params_wrapper(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success"})
        await webhook_service.register_conversation("call-456", executor)

        async with await _make_client(webhook_service) as client:
            response = await client.post(
                "/tools/call-456/update_reservation",
                json={"function_params": {"reservation_id": "ABC123", "seat": "12A"}},
            )

        assert response.status_code == 200
        assert response.json() == {"status": "success"}
        executor.execute.assert_awaited_once_with(
            "update_reservation",
            {"reservation_id": "ABC123", "seat": "12A"},
        )

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_conversation(self, webhook_service: ToolWebhookService):
        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/missing/get_reservation", json={"reservation_id": "ABC123"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown call_id: missing"

    @pytest.mark.asyncio
    async def test_rejects_invalid_function_params(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success"})
        await webhook_service.register_conversation("call-789", executor)

        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/call-789/get_reservation", json={"function_params": "bad"})

        assert response.status_code == 400
        assert response.json()["detail"] == "function_params must be a JSON object"
        executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_route_id_routes_by_session(self, webhook_service: ToolWebhookService):
        """call_session_id alias routes tool calls to the correct conversation."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "routed"})
        await webhook_service.register_conversation("record-1", executor)
        await webhook_service.register_route_id("session-abc-123", "record-1")

        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/session-abc-123/get_reservation", json={"id": "R1"})

        assert response.status_code == 200
        assert response.json() == {"status": "routed"}
        executor.execute.assert_awaited_once_with("get_reservation", {"id": "R1"})

    @pytest.mark.asyncio
    async def test_concurrent_sessions_route_independently(self, webhook_service: ToolWebhookService):
        """Two concurrent conversations with different session IDs route correctly."""
        executor_a = MagicMock()
        executor_a.execute = AsyncMock(return_value={"agent": "A"})
        executor_b = MagicMock()
        executor_b.execute = AsyncMock(return_value={"agent": "B"})

        await webhook_service.register_conversation("record-a", executor_a)
        await webhook_service.register_conversation("record-b", executor_b)
        await webhook_service.register_route_id("session-111", "record-a")
        await webhook_service.register_route_id("session-222", "record-b")

        async with await _make_client(webhook_service) as client:
            resp_a = await client.post("/tools/session-111/get_reservation", json={})
            resp_b = await client.post("/tools/session-222/get_reservation", json={})

        assert resp_a.json() == {"agent": "A"}
        assert resp_b.json() == {"agent": "B"}
        executor_a.execute.assert_awaited_once()
        executor_b.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unregister_removes_session_id_alias(self, webhook_service: ToolWebhookService):
        """Unregistering a conversation also removes call_session_id aliases."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"ok": True})
        await webhook_service.register_conversation("record-x", executor)
        await webhook_service.register_route_id("session-xyz", "record-x")

        # Both keys work before unregister
        async with await _make_client(webhook_service) as client:
            assert (await client.post("/tools/session-xyz/test_tool", json={})).status_code == 200

        await webhook_service.unregister_conversation("record-x")

        # Session ID alias should be gone too
        async with await _make_client(webhook_service) as client:
            assert (await client.post("/tools/session-xyz/test_tool", json={})).status_code == 404

    @pytest.mark.asyncio
    async def test_unregister_conversation_removes_audit_log(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success"})
        await webhook_service.register_conversation("call-000", executor)
        assert await webhook_service.get_audit_log("call-000") is not None

        await webhook_service.unregister_conversation("call-000")

        assert await webhook_service.get_audit_log("call-000") is None
