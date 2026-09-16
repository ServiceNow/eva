"""Per-tick diagnostic trace for the cascade caller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionLog:
    """Write and flush simulator diagnostics as they occur."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._handle: TextIO | None = None
        self._counts: dict[str, int] = {}

    def log(self, kind: str, **fields: Any) -> None:
        """Append one trace row of the given kind and flush it to disk."""
        self._counts[kind] = self._counts.get(kind, 0) + 1
        try:
            handle = self._open()
            handle.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")
            handle.flush()
        except Exception as exc:
            logger.warning(f"Could not write caller decision trace: {exc}")

    def _open(self) -> TextIO:
        """Open the trace on first use, so a run that traces nothing leaves no file."""
        if self._handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.output_path, "w")
        return self._handle

    def save(self) -> None:
        """Close the trace. Safe to call more than once."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            logger.info(f"Caller decision trace written to {self.output_path}")

    def summary(self) -> dict[str, int]:
        """Count rows by kind, for a one-line end-of-run report."""
        return dict(self._counts)
