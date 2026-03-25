"""FastAPI webhook service exposing EVA tools over HTTP."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _ConversationRegistration:
    """Webhook routing state for a single external conversation."""

    executor: ToolExecutor
    audit_log: AuditLog = field(default_factory=AuditLog)


class ToolWebhookService:
    """Expose per-conversation tool executors as HTTP webhook endpoints."""

    def __init__(self, port: int = 8888, host: str = "0.0.0.0"):
        self.port = port
        self.host = host

        self._lock = asyncio.Lock()
        self._conversations: dict[str, _ConversationRegistration] = {}

        self._app = FastAPI()
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._running = False

        self._register_routes()

    @property
    def app(self) -> FastAPI:
        """Return the FastAPI app for in-process tests."""
        return self._app

    async def start(self) -> None:
        """Start the webhook server."""
        if self._running:
            logger.warning("Tool webhook service already running")
            return

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        self._running = True

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Tool webhook service started on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the webhook server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.should_exit = True

        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass

        self._server = None
        self._server_task = None
        logger.info("Tool webhook service stopped")

    async def register_conversation(
        self,
        call_id: str,
        executor: ToolExecutor,
        audit_log: AuditLog | None = None,
    ) -> None:
        """Register a conversation call ID to route future webhook requests."""
        registration = _ConversationRegistration(executor=executor, audit_log=audit_log or AuditLog())
        async with self._lock:
            self._conversations[call_id] = registration
        logger.info(f"Registered tool webhook conversation {call_id}")

    async def unregister_conversation(self, call_id: str) -> None:
        """Remove a conversation from the webhook registry."""
        async with self._lock:
            self._conversations.pop(call_id, None)
        logger.info(f"Unregistered tool webhook conversation {call_id}")

    async def get_audit_log(self, call_id: str) -> AuditLog | None:
        """Return the audit log for a registered conversation."""
        async with self._lock:
            registration = self._conversations.get(call_id)
            return registration.audit_log if registration else None

    async def _get_registration(self, call_id: str) -> _ConversationRegistration | None:
        async with self._lock:
            return self._conversations.get(call_id)

    def _register_routes(self) -> None:
        @self._app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.post("/tools/{call_id}/{tool_name}")
        async def invoke_tool(call_id: str, tool_name: str, request: Request) -> Any:
            registration = await self._get_registration(call_id)
            if registration is None:
                raise HTTPException(status_code=404, detail=f"Unknown call_id: {call_id}")

            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON request body") from exc

            if body is None:
                params: dict[str, Any] = {}
            elif isinstance(body, dict):
                function_params = body.get("function_params")
                if function_params is None:
                    params = body
                elif isinstance(function_params, dict):
                    params = function_params
                else:
                    raise HTTPException(status_code=400, detail="function_params must be a JSON object")
            else:
                raise HTTPException(status_code=400, detail="Tool webhook body must be a JSON object")

            registration.audit_log.append_tool_call(tool_name, params)
            result = await registration.executor.execute(tool_name, params)
            registration.audit_log.append_tool_response(tool_name, result)

            logger.info(f"Executed webhook tool {tool_name} for call {call_id}")
            return result

        @self._app.websocket("/media-stream/{conversation_id}")
        async def media_stream(websocket: WebSocket, conversation_id: str) -> None:
            """Accept Telnyx Call Control media stream connections."""
            from eva.assistant.transports.call_control import get_active_transport

            transport = get_active_transport(conversation_id)
            if transport is None:
                logger.warning("No active transport for media stream conversation %s", conversation_id)
                await websocket.close(code=1008, reason="No active transport for this conversation")
                return

            await websocket.accept()
            logger.info("Accepted media stream WebSocket for conversation %s", conversation_id)
            try:
                await transport.handle_media_stream(websocket)
            except WebSocketDisconnect:
                logger.info("Media stream WebSocket disconnected for %s", conversation_id)
            except Exception as exc:
                logger.error("Media stream error for %s: %s", conversation_id, exc)
