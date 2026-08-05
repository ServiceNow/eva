"""Concrete ``BackendFactory`` mapping provider names to ``Backend`` classes.

Mirrors the lazy-import pattern of ``eva.user_simulator.factory`` and
``eva.orchestrator.worker._get_server_class``: each provider's SDK is imported
only when that provider is selected, so unused providers need not be
importable. The same factory serves either role -- a ``Backend`` has no role
knowledge (docs/refactor-step1.md).
"""

from __future__ import annotations

from typing import Any

from eva.backend.base import Backend
from eva.backend.factory import BackendFactory


class DefaultBackendFactory(BackendFactory):
    """Registry-backed factory for the built-in backends.

    Only ``openai_realtime`` is wired today; further providers are added here
    as their backends are migrated onto the ``Backend`` contract.
    """

    def create(self, name: str, config: dict[str, Any]) -> Backend:
        if name == "openai_realtime":
            from eva.backend.openai_realtime import OpenAIRealtimeBackend

            # The backend takes the flat config and assembles its own provider
            # session -- the caller doesn't hand-build any provider JSON.
            return OpenAIRealtimeBackend(config=config)
        raise ValueError(f"Unknown backend provider: {name!r}. Supported: openai_realtime")
