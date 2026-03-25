"""Assistant components for the voice agent benchmark framework."""

import importlib

from eva.assistant.base import AssistantServerBase

# Registry of available assistant frameworks.
# Each entry maps a framework name to the fully-qualified class path.
# Lazy imports prevent loading framework dependencies (e.g. Pipecat)
# when a different framework is selected.
_REGISTRY: dict[str, str] = {
    "pipecat": "eva.assistant.server.PipecatAssistantServer",
    "roomkit": "eva.assistant.roomkit_server.RoomKitAssistantServer",
}


def create_assistant_server(framework: str, **kwargs) -> AssistantServerBase:
    """Create an assistant server for the given framework.

    Args:
        framework: Framework identifier (e.g. ``"pipecat"``, ``"roomkit"``).
        **kwargs: Arguments forwarded to the server constructor.

    Returns:
        An ``AssistantServerBase`` implementation.

    Raises:
        ValueError: If the framework is not registered.
    """
    if framework not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown assistant framework '{framework}'. Available: {available}")

    module_path, class_name = _REGISTRY[framework].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
