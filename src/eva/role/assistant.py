"""``AssistantRole`` contract: the business-side answering role.

DESIGN ONLY (Step 1 of the refactor, see docs/refactor-step1.md). Method
bodies are stubs; nothing here is wired into the existing code path yet.

Plug-in point (where this will eventually replace existing code):
    Today the assistant side is a concrete ``AbstractAssistantServer``
    subclass selected by ``eva.orchestrator.worker._get_server_class(framework)``
    (worker.py) and constructed + started inside
    ``ConversationWorker._start_assistant()`` (worker.py), which calls
    ``server_cls(...).start()``. Its outputs are flushed by
    ``ConversationWorker._cleanup()`` via ``server.stop()`` (which internally
    calls ``save_outputs()``), and ``get_conversation_stats()`` /
    ``get_final_scenario_db()`` are read back in ``ConversationWorker.run()``.

    In a later phase, ``_start_assistant()`` becomes the construction site for
    an ``AssistantRole`` (framework string -> ``backend_name`` passed to the
    ``BackendFactory``), and the worker drives ``role.run()`` /
    ``role.save_outputs()`` / ``role.get_final_scenario_db()`` instead of the
    server's own lifecycle methods. The provider-specific server subclasses
    collapse into ``Backend`` implementations behind the factory; the
    role-agnostic orchestration in ``ConversationWorker`` stays put. This
    module is deliberately separate from ``eva.role.user`` so that migration
    can land assistant-side first without touching the user-side diff.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eva.backend.factory import BackendFactory
from eva.role.base import Role


class AssistantRole(Role):
    """Role that answers on behalf of the business (today's "assistant server").

    Carries agent configuration and tool catalog; owns a ``ToolExecutor``
    (constructed by subclasses, not by this contract) to fulfill
    ``handle_tool_call_request``.

    Self-nudge: if the caller goes quiet for too long mid-call, the assistant
    itself proactively re-engages ("are you still there?") rather than
    waiting forever -- this is assistant-initiated, unlike a caller nudging an
    unresponsive agent. Unlike the tool-call/idle-detection seams elsewhere in
    this contract, self-nudging needs no new ``Role`` method and no new
    ``Backend`` event type: the nudge is just an ordinary outbound turn that
    this role's backend produces on its own after
    ``self_nudge_timeout_seconds`` of inactivity, using the same
    ``system_prompt``/instructions already established at ``open()`` time
    (see ``Backend.open``'s ``config`` docstring). Whether the *other* side
    (a ``UserRole``) needs to do anything special upon receiving it, versus
    just treating it as an ordinary assistant turn through its existing
    ``run()`` loop, is left open -- see docs/refactor-step1.md discussion;
    nothing here requires ``UserRole`` changes to handle it correctly today.
    """

    def __init__(
        self,
        *,
        backend_factory: BackendFactory,
        backend_name: str,
        backend_config: dict[str, Any],
        agent_config_path: str,
        scenario_db_path: str,
        current_date_time: str,
        self_nudge_timeout_seconds: float | None = None,
    ) -> None:
        """See ``Role.__init__`` for the backend-construction args.

        Args:
            agent_config_path: Path to the agent YAML (role, instructions,
                tool schemas) -- mirrors ``AbstractAssistantServer.agent`` /
                ``agent_config_path``.
            scenario_db_path: Path to the per-record scenario database JSON
                consumed by tool execution -- mirrors
                ``AbstractAssistantServer.scenario_db_path``.
            current_date_time: Current date/time string threaded into both
                prompt construction and tool execution (mirrors existing
                ``current_date_time`` plumbing throughout the assistant
                stack).
            self_nudge_timeout_seconds: How long the assistant backend should
                wait without hearing from the caller before proactively
                speaking again, or ``None`` to disable self-nudging entirely.
                This is an ``AssistantRole``-level knob, not a
                ``BackendCapabilities`` flag (capabilities describe what a
                backend *can* do, statically; this is a per-run tuning
                value). Wiring it into the constructed ``self.backend``'s own
                config (via ``backend_config`` / ``Backend.open(config=...)``)
                is left to the concrete subclass's constructor, same as
                elsewhere in this contract -- a ``Role`` does not otherwise
                reach into backend config after construction. A backend with
                no notion of provider-driven idle timing (e.g. a thin
                end-to-end backend) may simply ignore this value.
        """
        super().__init__(backend_factory=backend_factory, backend_name=backend_name, backend_config=backend_config)
        self.agent_config_path = agent_config_path
        self.scenario_db_path = scenario_db_path
        self.current_date_time = current_date_time
        self.self_nudge_timeout_seconds = self_nudge_timeout_seconds

    @abstractmethod
    def get_final_scenario_db(self) -> dict[str, Any]:
        """Return the (possibly mutated) scenario database state, for metrics.

        Mirrors ``AbstractAssistantServer.get_final_scenario_db()``.
        """
        ...
