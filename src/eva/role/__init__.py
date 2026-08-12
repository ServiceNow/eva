"""Provider-agnostic ``Role`` abstraction (see ``docs/refactor-step1.md``).

A ``Role`` owns everything that was duplicated across the assistant provider
stacks: prompt construction, tool ownership, and agent-config data. Each
``Role`` holds exactly one ``eva.backend.Backend`` instance, created at runtime
via a ``eva.backend.BackendFactory``. Only the assistant side is on this path
for now; the user simulator stays on its legacy stack.
"""

from eva.role.assistant import AssistantRole
from eva.role.base import Role

__all__ = ["AssistantRole", "Role"]
