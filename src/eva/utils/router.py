"""LiteLLM Router — configure-once module.

Call `init(model_list)` once at startup, then `get()` anywhere to retrieve the shared Router instance.

Concurrency is not limited here: since LiteLLM 1.98 the Router rejects calls over a deployment's
``max_parallel_requests`` with an immediate 429 instead of queueing them, so no default cap is set.
EVA bounds load upstream with ``max_concurrent_conversations`` and ``max_concurrent_judges``.
"""

from litellm import Router

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_router: Router | None = None


def init(model_list: list) -> None:
    """Create and store the shared Router. Call once at startup."""
    global _router
    logger.info(f"Initializing LiteLLM Router with {len(model_list)} deployment(s)")
    _router = Router(model_list=model_list, num_retries=0, timeout=60)


def reset() -> None:
    """Reset the shared Router to None."""
    global _router
    _router = None


def get() -> Router:
    """Return the shared Router. Raises if `init()` was not called."""
    if _router is None:
        raise RuntimeError("Router not initialized — call eva.utils.router.init() first")
    return _router
