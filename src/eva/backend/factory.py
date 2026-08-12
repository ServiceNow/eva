"""Factory interface for constructing ``Backend`` instances by name.

DESIGN ONLY (Step 1 of the refactor). Mirrors the shape of today's
``eva.user_simulator.factory.create_user_simulator`` (lazy per-provider
imports keyed off config type) but is provider-and-role-agnostic: the same
factory is meant to be usable to build a backend for either an
``AssistantRole`` or a ``UserRole``, since a ``Backend`` has no role
knowledge (that's the whole point of the split -- see docs/refactor-step1.md,
"lets any backend act as either role").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eva.backend.base import Backend


class BackendFactory(ABC):
    """Constructs a ``Backend`` for a named provider from a config blob.

    A concrete implementation is expected to hold (or look up) a registry
    mapping provider name -> ``Backend`` subclass, analogous to today's
    ``create_user_simulator`` / assistant-server construction in
    ``orchestrator/runner.py``, and to import each provider module lazily so
    that unused providers' SDKs need not be installed/imported.
    """

    @abstractmethod
    def create(self, name: str, config: dict[str, Any]) -> Backend:
        """Construct and return a not-yet-opened ``Backend``.

        Args:
            name: Provider identifier (e.g. ``"openai_realtime"``,
                ``"gemini_live"``, ``"elevenlabs"``, ``"cascade"``). The set
                of valid names is defined by the concrete factory's registry,
                not by this interface.
            config: Provider-specific configuration understood by that
                backend's ``open()`` (see ``Backend.open``). This factory
                does not validate the shape of ``config`` beyond dispatching
                on ``name`` -- each ``Backend`` subclass is responsible for
                validating its own config.

        Returns:
            A constructed ``Backend`` instance. The returned backend has not
            had ``open()`` called on it yet -- construction and session
            establishment are separate steps so a ``Role`` can construct its
            backend early (e.g. at record setup) and open the session later
            (e.g. once the other party is ready).

        Raises:
            ValueError: if ``name`` does not match a known provider.
        """
        ...
