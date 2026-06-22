import logging
from typing import Any

import tiktoken


def approximate_reasoning_tokens(
    reasoning_content: str, model: str, warned_flag_holder: object, logger: logging.Logger
) -> int:
    """Approximate reasoning token count via tiktoken when the API response omits it.

    Tries the model-specific encoding first; falls back to cl100k_base for
    proprietary or unknown model names. Logs a one-time warning per caller
    instance (tracked via ``warned_flag_holder._reasoning_token_fallback_warned``).
    """
    if not getattr(warned_flag_holder, "_reasoning_token_fallback_warned", False):
        logger.warning(
            "No reasoning token count found in API response for model '%s'; "
            "falling back to tiktoken approximation. This warning will not repeat.",
            model,
        )
        warned_flag_holder._reasoning_token_fallback_warned = True
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(reasoning_content))


def _resolve_url(params: dict[str, Any], counter: int) -> tuple[str | None, int]:
    """Resolve a single URL from params, supporting round-robin across multiple URLs.

    If params contains a "urls" list, selects the next URL via round-robin and
    increments the counter. Otherwise falls back to the single "url" parameter.

    Args:
        params: Service parameters dict (may contain "url" or "urls").
        counter: Current round-robin counter value.

    Returns:
        Tuple of (selected_url, updated_counter).
    """
    urls = params.get("urls")
    if urls and isinstance(urls, list) and len(urls) > 0:
        selected = urls[counter % len(urls)]
        return selected, counter + 1
    return params.get("url"), counter
