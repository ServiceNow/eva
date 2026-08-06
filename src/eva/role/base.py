"""Abstract ``Role`` base contract: prompt/tools/goal ownership over a ``Backend``.

DESIGN ONLY (Step 1 of the refactor, see docs/refactor-step1.md). Every
method body here is a stub -- this module defines shapes, not behavior, and
is not imported by any existing code path.

This module holds only the shared ``Role`` base. The two concrete roles live
in sibling modules -- ``AssistantRole`` in ``eva.role.assistant`` and
``UserRole`` in ``eva.role.user`` -- since each carries a meaningfully
different data payload and plug-in point (see those modules' docstrings) and
is expected to grow its own implementation in later phases; keeping them in
separate files keeps each phase's diff scoped to one role.

Design choice -- one ``Role`` base with ``AssistantRole``/``UserRole``
subclasses, rather than two unrelated ABCs:
    Both roles share an identical *control loop* shape: construct a backend
    via ``BackendFactory``, ``build_prompt()`` before opening it, drive
    ``backend.receive()`` and dispatch tool-call requests to
    ``handle_tool_call_request()``, and record recorded audio/transcript for
    output. What differs between them is only the *data* they carry (agent
    config + tool catalog for the assistant; goal + persona + starting
    utterance for the user) and how they decide the conversation is over.
    That's a difference in constructor args and a couple of abstract methods,
    not in control flow -- so one shared base with two thin subclasses avoids
    duplicating the event loop, while still keeping tool-ownership and
    prompt-building role-specific via abstract methods. If the two roles'
    control loops diverge significantly in a later phase, splitting them
    apart is a mechanical extraction of ``Role`` into two ABCs -- nothing
    here should make that harder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from eva.backend.base import Backend, ToolCallRequest, ToolCallResult
from eva.backend.factory import BackendFactory


class Role(ABC):
    """Owns prompt, tools/goal, and a runtime-created ``Backend``.

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

    def __init__(self, *, backend_factory: BackendFactory, backend_name: str, backend_config: dict[str, Any]) -> None:
        """Construct the role's backend (but do not open its session yet).

        Args:
            backend_factory: Factory used to construct ``self.backend``.
            backend_name: Provider name passed through to
                ``BackendFactory.create``.
            backend_config: Provider-specific config passed through to
                ``BackendFactory.create`` (not to be confused with the
                ``config`` argument of ``Backend.open``, which is also
                provider-specific but may be augmented by the role at
                open-time, e.g. with a resolved sample rate).
        """
        self.backend: Backend = backend_factory.create(backend_name, backend_config)

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
    async def run(self) -> str:
        """Drive the conversation for this role until it reaches a terminal state.

        Expected shape (left to subclasses to implement, not prescribed in
        detail here since the exact loop depends on the backend's
        capabilities -- see ``BackendCapabilities``):
        1. ``await self.backend.open(system_prompt=self.build_prompt(), ...)``
        2. Iterate ``self.backend.receive()``, dispatching
           ``TOOL_CALL_REQUEST`` events to ``handle_tool_call_request`` and
           feeding the ``ToolCallResult`` back via
           ``self.backend.send(tool_result=...)``.
        3. Record audio/transcript events as they arrive (see
           ``record_audio``).
        4. On a terminal event (hangup, timeout, transfer, error), call
           ``await self.backend.close()`` and return an end-reason string.

        Returns:
            A short end-reason string (e.g. ``"goodbye"``, ``"transfer"``,
            ``"timeout"``, ``"error"``) -- mirrors the return contract of
            today's ``AbstractUserSimulator.run_conversation()``.
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

    @abstractmethod
    async def save_outputs(self, output_dir: Path) -> None:
        """Persist this role's output artifacts to ``output_dir``.

        For ``AssistantRole`` this covers ``audit_log.json``,
        ``transcript.jsonl``, scenario DB snapshots (mirrors
        ``AbstractAssistantServer.save_outputs``). For ``UserRole`` this
        covers ``user_simulator_events.jsonl`` (mirrors the event logger in
        ``AbstractUserSimulator``). Audio WAV files are expected to be
        written by the shared audio-recording helper referenced in
        ``record_audio``, not necessarily by this method -- exact division of
        labor is left to the later implementation phase.
        """
        ...
