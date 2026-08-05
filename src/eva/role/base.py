"""Abstract ``Role`` base contract: prompt/tools/goal ownership over a ``Backend``.

Step 1 of the refactor (see docs/refactor-step1.md). Abstract base shared by
the concrete ``AssistantRole`` / ``UserRole``, which the worker constructs
behind the ``USE_ROLE_BACKEND_OPENAI_REALTIME`` gate.

This module holds only the shared ``Role`` base. The two concrete roles live
in sibling modules -- ``AssistantRole`` in ``eva.role.assistant`` and
``UserRole`` in ``eva.role.user`` -- since each carries a meaningfully
different data payload and plug-in point (see those modules' docstrings) and
is expected to grow its own implementation in later phases; keeping them in
separate files keeps each phase's diff scoped to one role.

Design choice -- one ``Role`` base with ``AssistantRole``/``UserRole``
subclasses:
    The two roles share the *seams* that don't depend on transport direction:
    ``build_prompt()`` (instructions handed to the backend at open-time),
    ``handle_tool_call_request()`` (role-side tool execution), and
    ``record_audio()`` (accumulating audio for output). Those live here.

    Their *lifecycle* does differ today, and deliberately so: the assistant is
    a WebSocket **server** the user connects to (passive; ``start()`` /
    ``stop()``), while the user is the **driver** that dials in and runs the
    conversation to completion (``run()``). That asymmetry is a consequence of
    there being no mediator yet (docs/refactor-step1.md keeps ``Backend``
    direction-agnostic precisely so a later mediator can absorb the transport
    and re-symmetrize the two roles). Rather than force an ill-fitting uniform
    ``run()`` onto the server-shaped assistant, the lifecycle entry points live
    on the subclasses (``AssistantRole`` / ``UserRole``); the scaffold
    anticipated this ("if the two roles' control loops diverge, splitting is a
    mechanical extraction"). When the mediator lands, both sides can converge
    on a single driven loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from eva.backend.base import Backend, ToolCallRequest, ToolCallResult


class Role(ABC):
    """Owns prompt, tools/goal, and drives a worker-injected ``Backend``.

    A ``Role`` is the thing that used to be split across
    ``AbstractAssistantServer`` (assistant side) and ``AbstractUserSimulator``
    (user side): everything that is *not* pure provider API exchange lives
    here instead of in ``Backend``. In particular:

    - Tool execution stays role-side (per docs/refactor-step1.md): a ``Role``
      is responsible for turning a ``ToolCallRequest`` surfaced by its
      backend into a ``ToolCallResult``, using whatever execution engine is
      appropriate for that role (``ToolExecutor`` for ``AssistantRole``; a
      trivial/no-op handler for ``UserRole``, which today only exposes a
      synthetic ``end_call`` tool). ``Backend`` implementations never
      execute tools themselves.
    - Prompt construction stays role-side: ``build_prompt()`` replaces both
      ``AbstractAssistantServer._build_system_prompt()`` /
      ``AgenticSystem``'s prompt building and
      ``AbstractUserSimulator._build_prompt()``.
    - Audio recording / output persistence is a role-side concern shared by
      both subclasses (see ``docs/refactor-step1.md`` point 5, "consolidate
      audio recording / output-saving into one shared helper") -- this base
      class declares the seam (``record_audio`` / ``save_outputs``) but does
      not implement the shared helper itself; that helper is later work.
    """

    def __init__(self, *, backend: Backend) -> None:
        """Take the (not-yet-opened) backend the role will drive.

        The role does **not** construct its own backend and knows nothing about
        the ``BackendFactory``. The worker owns the factory, calls
        ``factory.create(name, config)``, and injects the resulting ``Backend``
        here. This keeps backend selection/configuration a worker concern and
        lets the same backend be wired to either role.

        Args:
            backend: A constructed, not-yet-opened ``Backend`` (see
                ``BackendFactory.create``). The role opens a session on it in
                ``run()`` and holds the returned ``BackendSession`` handle;
                per-exchange state lives on that handle, not on the backend.
        """
        self.backend = backend

    @abstractmethod
    def build_prompt(self) -> str:
        """Build the full system prompt / instructions for this role.

        For ``AssistantRole`` this replaces
        ``AbstractAssistantServer._build_system_prompt()`` and
        ``AgenticSystem``'s inline prompt construction. For ``UserRole`` this
        replaces ``AbstractUserSimulator._build_prompt()``. Called before
        ``Backend.open()`` so the result can be passed as its
        ``system_prompt`` argument.
        """
        ...

    @abstractmethod
    async def handle_tool_call_request(self, request: ToolCallRequest) -> ToolCallResult:
        """Execute a tool call surfaced by this role's backend and return the result.

        This is the single place tool execution happens for this role --
        ``Backend`` implementations must never execute tools directly (see
        class docstring). Implementations should log the call/result (e.g.
        to an audit log) as part of executing it.
        """
        ...

    @abstractmethod
    def record_audio(self, source: str, audio_data: bytes) -> None:
        """Accumulate a chunk of audio for later persistence.

        Args:
            source: Role-defined stream label (e.g. ``"user"``,
                ``"assistant"``, or a cleaned/pre-perturbation variant).
                Mirrors ``AbstractUserSimulator._record_audio`` /
                ``AbstractAssistantServer``'s audio-buffer fields; the exact
                set of valid labels is left to subclasses/shared helper, not
                fixed by this contract.
            audio_data: Raw PCM16 bytes at this role's recording sample rate.
        """
        ...
