"""LiteLLM Router — configure-once module.

Call `init(model_list)` once at startup, then `get()` anywhere to retrieve the shared Router instance.
"""

from litellm import Router

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_router: Router | None = None


def init(model_list: list) -> None:
    """Create and store the shared Router. Call once at startup."""
    global _router
    logger.info(f"Initializing LiteLLM Router with {len(model_list)} deployment(s)")
    _router = Router(model_list=model_list, num_retries=0, default_max_parallel_requests=5, timeout=60)


def reset() -> None:
    """Reset the shared Router to None."""
    global _router
    _router = None


def get() -> Router:
    """Return the shared Router. Raises if `init()` was not called."""
    if _router is None:
        raise RuntimeError("Router not initialized — call eva.utils.router.init() first")
    return _router
