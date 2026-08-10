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

    Turn-end fallback (self-nudge): the assistant's backstop for a *dropped
    user turn*. When VAD / turn detection silently fails to fire for a real
    user utterance, the call would otherwise hang until the provider's
    inactivity timeout ends it. After the assistant stops speaking, if no user
    turn is detected within ``turn_end_fallback_seconds``, the assistant
    proactively re-engages with a nudge (acknowledge-and-answer if partial
    user speech/audio was captured, otherwise ask the caller to repeat). This
    is the seam already shipped as the pipeline-side ``TurnEndFallbackTimer``
    (see ``eva.assistant.pipeline.fallback`` and ``EVA_TURN_END_FALLBACK_TIME``);
    it works for both cascade and audio-LLM pipelines.

    Two policies the backend owns, carried over from the shipped feature:
    - Give up after a small number of *consecutive* nudges without a real user
      turn resetting the count (``MAX_CONSECUTIVE_FALLBACK_NUDGES``), then let
      the provider's inactivity backstop end the call.
    - Never nudge once the call is ending (a nudge during teardown produces a
      phantom assistant turn after the conversation is logically closed).

    Unlike the tool-call/idle-detection seams elsewhere in this contract, the
    fallback needs no new ``Role`` method and no new ``Backend`` event type:
    the nudge is just an ordinary outbound turn that this role's backend
    produces on its own after the timeout, using the same
    ``system_prompt``/instructions already established at ``open()`` time (see
    ``Backend.open``'s ``config`` docstring). It is surfaced through the normal
    ``receive()`` stream and tagged so downstream metrics can identify and zero
    it (the shipped feature records the transcript marker with
    ``message_type="turn_fallback"``; see ``BackendEvent.metadata``). Whether
    the *other* side (a ``UserRole``) needs to do anything special upon
    receiving it, versus just treating it as an ordinary assistant turn through
    its existing ``run()`` loop, is left open -- see docs/refactor-step1.md
    discussion; nothing here requires ``UserRole`` changes to handle it today.
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
        turn_end_fallback_seconds: float | None = None,
    ) -> None:
        """Initialize the assistant role.

        Args:
            backend_factory: Factory used to construct the backend.
            backend_name: Key passed to the factory to select a backend.
            backend_config: Provider-specific configuration for the backend.
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
            turn_end_fallback_seconds: How long after the assistant stops
                speaking to wait for a user turn before firing a turn-end
                fallback nudge, or ``None`` to disable the fallback entirely
                (preserving the old behavior of waiting for the provider's
                inactivity timeout). Mirrors the shipped
                ``EVA_TURN_END_FALLBACK_TIME`` knob. This is an
                ``AssistantRole``-level tuning value, not a
                ``BackendCapabilities`` flag (capabilities describe what a
                backend *can* do, statically). Wiring it into the constructed
                ``self.backend``'s own config (via ``backend_config`` /
                ``Backend.open(config=...)``) is left to the concrete
                subclass's constructor, same as elsewhere in this contract --
                a ``Role`` does not otherwise reach into backend config after
                construction. A backend with no notion of idle timing (e.g. a
                thin end-to-end backend that relies on its own provider
                backstop) may simply ignore this value.
        """
        super().__init__(backend_factory=backend_factory, backend_name=backend_name, backend_config=backend_config)
        self.agent_config_path = agent_config_path
        self.scenario_db_path = scenario_db_path
        self.current_date_time = current_date_time
        self.turn_end_fallback_seconds = turn_end_fallback_seconds

    @abstractmethod
    def get_final_scenario_db(self) -> dict[str, Any]:
        """Return the (possibly mutated) scenario database state, for metrics.

        Mirrors ``AbstractAssistantServer.get_final_scenario_db()``.
        """
        ...
