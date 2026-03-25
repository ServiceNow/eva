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
        # Track unique registrations (by id) vs aliases that point to the same one
        self._unique_registrations: set[int] = set()

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
            self._unique_registrations.add(id(registration))
        logger.info(f"Registered tool webhook conversation {call_id}")

    async def add_alias(self, alias: str, primary_call_id: str) -> None:
        """Register an additional key that routes to the same conversation as primary_call_id."""
        async with self._lock:
            registration = self._conversations.get(primary_call_id)
            if registration is None:
                logger.warning("Cannot alias %s → %s: primary not found", alias, primary_call_id)
                return
            self._conversations[alias] = registration
        logger.info("Registered tool webhook alias %s → %s", alias, primary_call_id)

    async def unregister_conversation(self, call_id: str) -> None:
        """Remove a conversation (and all its aliases) from the webhook registry."""
        async with self._lock:
            registration = self._conversations.pop(call_id, None)
            if registration is not None:
                reg_id = id(registration)
                # Remove all aliases pointing to the same registration
                aliases = [k for k, v in self._conversations.items() if id(v) == reg_id]
                for alias in aliases:
                    del self._conversations[alias]
                self._unique_registrations.discard(reg_id)
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

        @self._app.post("/call-control-events")
        async def call_control_events(request: Request) -> dict[str, str]:
            """Acknowledge Telnyx Call Control webhook events (answer, hangup, etc.)."""
            return {"status": "ok"}

        @self._app.post("/tools/{call_id}/{tool_name}")
        async def invoke_tool(call_id: str, tool_name: str, request: Request) -> Any:
            registration = await self._get_registration(call_id)
            if registration is None:
                # Try URL-decoded version (call_control_ids may contain colons)
                from urllib.parse import unquote
                registration = await self._get_registration(unquote(call_id))
            if registration is None:
                # Fallback: {{call_control_id}} in tool URLs resolves to the
                # assistant's B-leg CC ID, which we don't know in advance.
                # If there's exactly one unique conversation, route there.
                async with self._lock:
                    unique_count = len(self._unique_registrations)
                    if unique_count == 1:
                        registration = next(iter(self._conversations.values()))
                        # Auto-register this B-leg CC ID so subsequent calls are fast
                        self._conversations[call_id] = registration
                if registration is not None:
                    logger.info(
                        "Auto-registered unknown call_id %s (B-leg) → sole active conversation",
                        call_id,
                    )
            if registration is None:
                logger.warning("Tool webhook 404: call_id=%s, active conversations=%d", call_id, len(self._conversations))
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
