"""LiteLLM Router — configure-once module.

Call `init(model_list)` once at startup, then `get()` anywhere to retrieve the shared Router instance.

Concurrency: since LiteLLM 1.98 the Router rejects calls over a deployment's ``max_parallel_requests`` with an
immediate 429 instead of queueing them. EVA relies on queueing, so it bounds concurrency itself: every call goes
through `limit(model_group)`, which waits for a free slot and yields the specific deployment to call. Callers pass
that deployment's id as ``model`` so the Router sends the call there, rather than picking a deployment on its own
that may already be at its cap.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from litellm import Router
from litellm.utils import calculate_max_parallel_requests

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_PARALLEL_REQUESTS = 5


@dataclass
class Deployment:
    """One entry of EVA_MODEL_LIST. Pass `id` as the Router's ``model`` to call exactly this deployment."""

    id: str
    litellm_params: dict[str, Any] = field(default_factory=dict)
    capacity: int = _DEFAULT_MAX_PARALLEL_REQUESTS
    in_use: int = 0


@dataclass
class _ModelGroup:
    deployments: list[Deployment]
    semaphore: asyncio.Semaphore


_router: Router | None = None
_groups: dict[str, _ModelGroup] = {}


def init(model_list: list) -> None:
    """Create and store the shared Router. Call once at startup."""
    global _router, _groups
    logger.info(f"Initializing LiteLLM Router with {len(model_list)} deployment(s)")
    _router = Router(
        model_list=model_list,
        num_retries=0,
        default_max_parallel_requests=_DEFAULT_MAX_PARALLEL_REQUESTS,
        timeout=60,
    )
    # The Router assigns each deployment a model_info.id, so read the deployments back from it.
    _groups = _build_groups(_router.model_list)


def _build_groups(model_list: list) -> dict[str, _ModelGroup]:
    """Group deployments by model_name, each with its own concurrency cap and a semaphore for the group total."""
    deployments: dict[str, list[Deployment]] = {}
    for deployment in model_list:
        # TODO: max_parallel_requests/rpm/tpm are read from litellm_params for backwards compatibility with
        # existing EVA_MODEL_LIST configs. This is now an EVA-level setting (LiteLLM no longer queues), so it
        # should move out of litellm_params to the top level of the deployment, next to use_responses_api.
        # TODO: evaluate whether per-deployment caps are needed at all, since max_concurrent_conversations
        # already bounds concurrency.
        params = deployment.get("litellm_params", {})
        cap = calculate_max_parallel_requests(
            max_parallel_requests=params.get("max_parallel_requests"),
            rpm=params.get("rpm"),
            tpm=params.get("tpm"),
            default_max_parallel_requests=_DEFAULT_MAX_PARALLEL_REQUESTS,
        )
        deployments.setdefault(deployment["model_name"], []).append(
            Deployment(id=deployment["model_info"]["id"], litellm_params=params, capacity=cap)
        )
    return {
        name: _ModelGroup(group, asyncio.Semaphore(sum(d.capacity for d in group)))
        for name, group in deployments.items()
    }


def reset() -> None:
    """Reset the shared Router to None."""
    global _router, _groups
    _router = None
    _groups = {}


def get() -> Router:
    """Return the shared Router. Raises if `init()` was not called."""
    if _router is None:
        raise RuntimeError("Router not initialized — call eva.utils.router.init() first")
    return _router


@asynccontextmanager
async def limit(model_group: str) -> AsyncIterator[Deployment]:
    """Hold a concurrency slot on one of `model_group`'s deployments, waiting if all slots are in use.

    Yields the deployment holding the slot; call the Router with ``model=deployment.id``. Hold the slot for the
    whole call, including consuming a stream. Groups with no configured deployment are unbounded and yield a
    placeholder whose id is the group name itself.
    """
    group = _groups.get(model_group)
    if group is None:
        yield Deployment(id=model_group)
        return
    async with group.semaphore:
        # The group semaphore is the sum of the caps, so some deployment always has a free slot here.
        deployment = min(group.deployments, key=lambda d: d.in_use / d.capacity)
        deployment.in_use += 1
        try:
            yield deployment
        finally:
            deployment.in_use -= 1
