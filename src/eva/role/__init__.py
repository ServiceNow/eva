"""Provider-agnostic ``Role`` abstraction (design-only, Step 1 of the refactor).

A ``Role`` owns everything that today is duplicated across the assistant and
user-simulator stacks per-provider: prompt construction, tool ownership, and
goal/persona/agent-config data. Each ``Role`` holds exactly one
``eva.backend.Backend`` instance, created at runtime via a
``eva.backend.BackendFactory``.

Nothing in this package is wired into the existing ``eva.assistant`` /
``eva.user_simulator`` code yet.
"""

from eva.role.assistant import AssistantRole
from eva.role.base import Role
from eva.role.user import UserRole

__all__ = ["AssistantRole", "Role", "UserRole"]
