"""Transcript accumulation for the caller's speech-to-text."""

from __future__ import annotations


class TranscriptBuffer:
    """Accumulates committed transcript segments separately from the in-flight partial."""

    def __init__(self) -> None:
        self.committed = ""
        self.in_flight = ""

    def apply_partial(self, text: str) -> None:
        """Replace the in-flight partial with the latest partial transcript."""
        self.in_flight = text

    def commit(self, text: str) -> None:
        """Append a finalized segment to the committed text and clear the partial."""
        self.committed = f"{self.committed} {text}".strip() if self.committed else text
        self.in_flight = ""

    def take_committed(self) -> str:
        """Return and clear the committed text."""
        text = self.committed
        self.committed = ""
        return text
