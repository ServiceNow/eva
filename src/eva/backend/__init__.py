"""Provider-agnostic ``Backend`` abstraction (see ``docs/refactor-step1.md``).

This package defines pure API/session objects (``Backend``) that know nothing
about role (assistant vs. user), plus a ``BackendFactory`` to construct them.
The worker builds a backend per conversation and drives it through an
``AssistantRole`` / ``UserRole`` for every provider the factory supports.
"""

from eva.backend.base import (
    Backend,
    BackendEvent,
    BackendEventType,
    BackendSession,
    ToolCallRequest,
    ToolCallResult,
)
from eva.backend.capabilities import BackendCapabilities
from eva.backend.factory import BackendFactory

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendEvent",
    "BackendEventType",
    "BackendFactory",
    "BackendSession",
    "ToolCallRequest",
    "ToolCallResult",
]
