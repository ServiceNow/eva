"""Provider-agnostic ``Backend`` abstraction (design-only, Step 1 of the refactor).

This package defines the contracts described in ``docs/refactor-step1.md``:
pure API/session objects (``Backend``) that know nothing about role
(assistant vs. user), plus a factory to construct them. Nothing in this
package is wired into the existing ``eva.assistant`` / ``eva.user_simulator``
code yet -- these are new, additive, currently-unused types.
"""

from eva.backend.base import Backend, BackendEvent, BackendEventType, ToolCallRequest, ToolCallResult
from eva.backend.capabilities import BackendCapabilities
from eva.backend.factory import BackendFactory

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendEvent",
    "BackendEventType",
    "BackendFactory",
    "ToolCallRequest",
    "ToolCallResult",
]
