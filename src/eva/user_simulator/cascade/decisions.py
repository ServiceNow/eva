"""Listener-reaction checks: should the caller interrupt or backchannel right now."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionLLM(Protocol):
    """Minimal interface the checks need from a model client."""

    async def decide(self, prompt: str) -> str:
        """Return the model's raw reply to a YES/NO question."""
        ...


@dataclass(frozen=True)
class CheckTrace:
    """What one YES/NO check actually did, for the diagnostic trace."""

    ran: bool
    raw: str = ""
    latency_ms: int = 0
    error: str = ""


@dataclass(frozen=True)
class ListenerVerdict:
    """Outcome of one check tick."""

    should_interrupt: bool
    should_backchannel: bool
    interrupt_trace: CheckTrace = field(default_factory=lambda: CheckTrace(ran=False))
    backchannel_trace: CheckTrace = field(default_factory=lambda: CheckTrace(ran=False))


def parse_yes_no(raw: str) -> bool:
    """Read a bare YES/NO reply. Anything unrecognized counts as NO."""
    return raw.strip().upper() == "YES"


class ListenerDecisions:
    """Runs the interrupt and backchannel checks concurrently against the partial transcript.

    Both fail closed: an exception yields "don't act", so a provider hiccup can
    never inject a barge-in that the caller never actually decided to make.
    """

    def __init__(
        self, llm: DecisionLLM, *, interrupt_prompt: str, backchannel_prompt: str, user_goal: str = ""
    ) -> None:
        self._llm = llm
        self._interrupt_prompt = interrupt_prompt
        self._backchannel_prompt = backchannel_prompt
        self._user_goal = user_goal

    async def evaluate(
        self, conversation_history: str, *, allow_interrupt: bool, allow_backchannel: bool
    ) -> ListenerVerdict:
        """Run whichever checks are enabled. Interrupt wins ties (tau: streaming.py:2549)."""
        interrupt_trace, backchannel_trace = await asyncio.gather(
            self._check(self._interrupt_prompt, conversation_history, enabled=allow_interrupt),
            self._check(self._backchannel_prompt, conversation_history, enabled=allow_backchannel),
        )
        interrupt = interrupt_trace.ran and parse_yes_no(interrupt_trace.raw)
        backchannel = backchannel_trace.ran and parse_yes_no(backchannel_trace.raw)
        return ListenerVerdict(
            should_interrupt=interrupt,
            should_backchannel=backchannel and not interrupt,
            interrupt_trace=interrupt_trace,
            backchannel_trace=backchannel_trace,
        )

    async def _check(self, template: str, conversation_history: str, *, enabled: bool) -> CheckTrace:
        """Ask the model one YES/NO question, reporting what happened rather than just the answer.

        Both templates are filled with the same arguments; `str.format` ignores the ones a
        given prompt does not use, so the backchannel prompt needs no goal slot.
        """
        if not enabled:
            return CheckTrace(ran=False)
        started = time.monotonic()
        try:
            filled = template.format(conversation_history=conversation_history, user_goal=self._user_goal)
            reply = await self._llm.decide(filled)
        except Exception as exc:
            logger.warning(f"Listener check failed, defaulting to no action: {exc}")
            return CheckTrace(ran=True, latency_ms=int((time.monotonic() - started) * 1000), error=str(exc))
        return CheckTrace(ran=True, raw=reply, latency_ms=int((time.monotonic() - started) * 1000))
