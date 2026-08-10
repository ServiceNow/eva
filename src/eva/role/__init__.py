"""Provider-agnostic ``Role`` abstraction (see ``docs/refactor-step1.md``).

A ``Role`` owns everything that was duplicated across the assistant and
user-simulator stacks per-provider: prompt construction, tool ownership, and
goal/persona/agent-config data. Each ``Role`` holds exactly one
``eva.backend.Backend`` instance, created at runtime via a
``eva.backend.BackendFactory``.
"""

from eva.role.assistant import AssistantRole
from eva.role.base import Role
from eva.role.user import UserRole

__all__ = ["AssistantRole", "Role", "UserRole"]
