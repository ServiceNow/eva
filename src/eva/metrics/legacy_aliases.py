"""Backwards-compatibility aliases for renamed metrics.

When a run produced before a rename is loaded (via ``--run-id``), old metric names may
appear in ``config.json`` (metrics list / validation_thresholds) and in per-record
``metrics.json`` payloads. Remap them at load-time so the rest of the code only sees the
new names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

LEGACY_METRIC_ALIASES: dict[str, str] = {
    "conversation_finished": "conversation_valid_end",
    "agent_turn_response": "conversation_correctly_finished",
}

_V = TypeVar("_V")


def rename_metric_keys(d: Mapping[str, _V]) -> dict[str, _V]:
    """Return a new dict with legacy metric names replaced by current names."""
    return {LEGACY_METRIC_ALIASES.get(k, k): v for k, v in d.items()}


def rename_metric_list(names: Iterable[str]) -> list[str]:
    """Return a list of metric names with legacy names replaced by current names."""
    return [LEGACY_METRIC_ALIASES.get(n, n) for n in names]
