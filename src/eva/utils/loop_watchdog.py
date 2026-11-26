"""Diagnostic: dump every thread's stack when the asyncio event loop stops running.

All live conversations share one event loop. When it blocks, every call stalls, and a
long enough block closes the local websockets for all of them at once. Set
``EVA_LOOP_WATCHDOG_S`` to a number of seconds (e.g. ``2``) to find out what blocked:

- A heartbeat on the loop re-arms ``faulthandler.dump_traceback_later`` every
  ``HEARTBEAT_S``. If the loop misses heartbeats for the threshold, faulthandler's own
  C thread writes all thread stacks to stderr. It does not need the GIL, so it still
  reports if a thread holds the GIL through the block.
- When the loop recovers, a warning gives how late the heartbeat was (a lower
  bound on the block, short by up to ``HEARTBEAT_S``), the process CPU time and
  the garbage-collection time during it. CPU close to the block's length means
  something was computing; CPU close to zero means waiting (I/O, a lock, or the
  process not being scheduled).

Off unless the variable is set. faulthandler supports one pending dump per process, so
only one watchdog should run at a time.
"""

import asyncio
import faulthandler
import gc
import os
import sys
import time
from typing import Self, TextIO

from eva.utils.logging import get_logger

ENV_VAR = "EVA_LOOP_WATCHDOG_S"
HEARTBEAT_S = 0.25

logger = get_logger(__name__)


class LoopWatchdog:
    """Context manager that watches the running event loop; see module docstring."""

    def __init__(self, threshold_s: float, file: TextIO | None = None, heartbeat_s: float = HEARTBEAT_S):
        if threshold_s <= heartbeat_s:
            raise ValueError(f"threshold_s ({threshold_s}) must be larger than heartbeat_s ({heartbeat_s})")
        self.threshold_s = threshold_s
        self.heartbeat_s = heartbeat_s
        self._file = file or sys.stderr
        self._handle: asyncio.TimerHandle | None = None
        self._gc_seconds = 0.0
        self._gc_started: float | None = None

    @classmethod
    def from_env(cls) -> "LoopWatchdog | None":
        """Watchdog configured from ``EVA_LOOP_WATCHDOG_S``, or None when unset or 0."""
        raw = os.environ.get(ENV_VAR, "").strip()
        if not raw:
            return None
        try:
            threshold = float(raw)
        except ValueError:
            raise ValueError(f"{ENV_VAR} must be a number of seconds, got {raw!r}") from None
        return cls(threshold) if threshold > 0 else None

    def __enter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        gc.callbacks.append(self._on_gc)
        print(
            f"Loop watchdog on: dumping all thread stacks to stderr when the event loop "
            f"is blocked for {self.threshold_s:g}s ({ENV_VAR})",
            file=self._file,
            flush=True,
        )
        self._last_tick = time.monotonic()
        self._last_cpu = time.process_time()
        self._last_gc = 0.0
        self._arm()
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        faulthandler.cancel_dump_traceback_later()
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)

    def _arm(self) -> None:
        # Re-arming cancels the pending dump, so it only fires if the next ticks never come.
        faulthandler.dump_traceback_later(self.threshold_s, repeat=False, file=self._file)
        self._handle = self._loop.call_later(self.heartbeat_s, self._tick)

    def _tick(self) -> None:
        now, cpu, gc_total = time.monotonic(), time.process_time(), self._gc_seconds
        blocked = now - self._last_tick - self.heartbeat_s
        if blocked >= self.threshold_s:
            logger.warning(
                f"Event loop was blocked for at least {blocked:.1f}s "
                f"(process CPU {cpu - self._last_cpu:.1f}s, GC {gc_total - self._last_gc:.1f}s); "
                f"thread stacks from {self.threshold_s:g}s in are on stderr"
            )
        self._last_tick, self._last_cpu, self._last_gc = now, cpu, gc_total
        self._arm()

    def _on_gc(self, phase: str, _info: dict) -> None:
        if phase == "start":
            self._gc_started = time.perf_counter()
        elif self._gc_started is not None:
            self._gc_seconds += time.perf_counter() - self._gc_started
            self._gc_started = None
