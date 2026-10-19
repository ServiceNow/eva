"""LiteLLM Router — configure-once module.

Call `init(model_list)` once at startup, then `get()` anywhere to retrieve the shared Router instance.

Concurrency: since LiteLLM 1.98 the Router rejects calls over a deployment's ``max_parallel_requests`` with an
immediate 429 instead of queueing them. EVA relies on queueing, so it bounds concurrency itself: every call goes
through `limit(model_group)` (or `acompletion`), which waits on a per-model-group semaphore.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from litellm import Router
from litellm.utils import calculate_max_parallel_requests

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_PARALLEL_REQUESTS = 5

_router: Router | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}


def init(model_list: list) -> None:
    """Create and store the shared Router. Call once at startup."""
    global _router, _semaphores
    logger.info(f"Initializing LiteLLM Router with {len(model_list)} deployment(s)")
    _router = Router(
        model_list=model_list,
        num_retries=0,
        default_max_parallel_requests=_DEFAULT_MAX_PARALLEL_REQUESTS,
        timeout=60,
    )
    _semaphores = _build_semaphores(model_list)


def _build_semaphores(model_list: list) -> dict[str, asyncio.Semaphore]:
    """One semaphore per model group, sized to the sum of its deployments' concurrency caps."""
    capacities: dict[str, int] = {}
    for deployment in model_list:
        # TODO: max_parallel_requests/rpm/tpm are read from litellm_params for backwards compatibility with
        # existing EVA_MODEL_LIST configs. This is now an EVA-level setting (LiteLLM no longer queues), so it
        # should move out of litellm_params to the top level of the deployment, next to use_responses_api.
        params = deployment.get("litellm_params", {})
        cap = calculate_max_parallel_requests(
            max_parallel_requests=params.get("max_parallel_requests"),
            rpm=params.get("rpm"),
            tpm=params.get("tpm"),
            default_max_parallel_requests=_DEFAULT_MAX_PARALLEL_REQUESTS,
        )
        name = deployment["model_name"]
        capacities[name] = capacities.get(name, 0) + cap
    return {name: asyncio.Semaphore(cap) for name, cap in capacities.items()}


def reset() -> None:
    """Reset the shared Router to None."""
    global _router, _semaphores
    _router = None
    _semaphores = {}


def get() -> Router:
    """Return the shared Router. Raises if `init()` was not called."""
    if _router is None:
        raise RuntimeError("Router not initialized — call eva.utils.router.init() first")
    return _router


@asynccontextmanager
async def limit(model_group: str) -> AsyncIterator[None]:
    """Hold a concurrency slot for `model_group`, waiting if all slots are in use.

    Hold it for the whole call, including consuming a stream. Groups with no configured deployment are unbounded.
    """
    semaphore = _semaphores.get(model_group)
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield


async def acompletion(**kwargs: Any) -> Any:
    """`Router.acompletion` bounded by the model group's concurrency limit (non-streaming calls)."""
    async with limit(kwargs["model"]):
        return await get().acompletion(**kwargs)
