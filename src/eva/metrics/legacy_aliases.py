"""Backwards-compatibility aliases for metric names renamed after runs were produced."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

LEGACY_METRIC_ALIASES: dict[str, str] = {
    "conversation_finished": "conversation_valid_end",
}

_V = TypeVar("_V")


def rename_metric_keys(d: Mapping[str, _V]) -> dict[str, _V]:
    return {LEGACY_METRIC_ALIASES.get(k, k): v for k, v in d.items()}


def rename_metric_list(names: Iterable[str]) -> list[str]:
    return [LEGACY_METRIC_ALIASES.get(n, n) for n in names]
