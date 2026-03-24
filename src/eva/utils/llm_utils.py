from typing import Any


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
