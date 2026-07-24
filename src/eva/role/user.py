"""``UserRole`` contract: the simulated-caller role.

DESIGN ONLY (Step 1 of the refactor, see docs/refactor-step1.md). Method
bodies are stubs; nothing here is wired into the existing code path yet.

Plug-in point (where this will eventually replace existing code):
    Today the user side is a concrete ``AbstractUserSimulator`` subclass
    selected by ``eva.user_simulator.factory.create_user_simulator(config, ...)``
    and constructed inside ``ConversationWorker._start_user_simulator()``
    (worker.py), which passes it ``server_url=f"ws://localhost:{port}/ws"``
    to reach the assistant server. The conversation is driven by
    ``ConversationWorker._run_conversation()`` calling
    ``user_simulator.run_conversation()``, whose returned end-reason string
    becomes the conversation result.

    In a later phase, ``_start_user_simulator()`` becomes the construction
    site for a ``UserRole`` (simulator config -> ``backend_name`` +
    ``backend_config`` for the ``BackendFactory``), and the worker drives
    ``role.run()`` (returning the same end-reason string via
    ``get_end_reason()``) instead of ``run_conversation()``. Note the
    ``server_url`` handoff is a *transport* detail that today's user side owns
    directly; per docs/refactor-step1.md the ``Backend`` contract is kept
    direction-agnostic precisely so this WS-connect concern can move into a
    ``Backend`` implementation (or, later, a mediator) without the role
    caring. This module is deliberately separate from ``eva.role.assistant``
    so the user-side migration can land as its own scoped diff.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from eva.backend.factory import BackendFactory
from eva.role.base import Role


class UserRole(Role):
    """Role that simulates the human caller (today's "user simulator").

    Carries goal/persona instead of agent config/tools -- its tool surface,
    if any, is limited to caller-side affordances like ``end_call`` (see
    ``END_CALL_DESCRIPTION`` in today's ``eva.user_simulator.base``), not a
    business tool catalog.
    """

    def __init__(
        self,
        *,
        backend_factory: BackendFactory,
        backend_name: str,
        backend_config: dict[str, Any],
        goal: dict[str, Any],
        persona_config: dict[str, Any],
        current_date_time: str,
    ) -> None:
        """Initialize the user role.

        Args:
            backend_factory: Factory used to construct the backend.
            backend_name: Key passed to the factory to select a backend.
            backend_config: Provider-specific configuration for the backend.
            goal: User goal / decision-tree data -- mirrors
                ``AbstractUserSimulator.goal``.
            persona_config: Persona/voice/behavior configuration -- mirrors
                ``AbstractUserSimulator.persona_config``.
            current_date_time: Threaded into prompt construction, mirroring
                existing plumbing.
        """
        super().__init__(backend_factory=backend_factory, backend_name=backend_name, backend_config=backend_config)
        self.goal = goal
        self.persona_config = persona_config
        self.current_date_time = current_date_time

    @abstractmethod
    def get_end_reason(self) -> str:
        """Return the terminal end-reason for this conversation.

        Mirrors the return value of today's
        ``AbstractUserSimulator.run_conversation()`` (``"goodbye"``,
        ``"transfer"``, ``"timeout"``, ``"error"``, ...).
        """
        ...
