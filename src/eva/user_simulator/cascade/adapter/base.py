"""Adapter contract: the only component doing real I/O against the assistant."""

from __future__ import annotations

from abc import ABC, abstractmethod

from eva.user_simulator.cascade.tick_result import TickResult


class Adapter(ABC):
    """Exchanges exactly one tick of audio with the assistant per call.

    The scheduler and simulator never learn which implementation is behind this
    interface, which is what lets a tick-driven transport replace the real-time
    one without touching turn-taking logic.
    """

    @abstractmethod
    async def start(self) -> None:
        """Establish the connection. Must return once ready to exchange audio."""

    @abstractmethod
    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None) -> TickResult:
        """Send this tick's caller audio (None means silence) and collect what arrived."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear the connection down. Must be safe to call twice."""
