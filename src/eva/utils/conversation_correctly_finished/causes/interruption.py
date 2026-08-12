"""Cause: a real accepted user turn was cancelled by an interruption and never resumed (``ended_with_user_interruption``)."""

import re
from typing import Any

from eva.utils.conversation_correctly_finished.signals import Classification, ConvFinishSignals
from eva.utils.log_processing import parse_log_message

_RE_INTR = re.compile(r"Interruption received - cancelling ongoing")
_RE_PROC = re.compile(r"Processing complete user turn: (.+)$")


def extract_log_signals(lines: list[str], resp_idx: list[int], s: ConvFinishSignals) -> None:
    """Interruption: cancelled a real accepted turn, with no response after the last interruption."""
    intr_idx = [i for i, ln in enumerate(lines) if _RE_INTR.search(ln)]
    if not intr_idx:
        return
    last_intr = intr_idx[-1]
    s.num_interruptions = len(intr_idx)
    proc = [(i, m.group(1).strip()) for i, ln in enumerate(lines) if (m := _RE_PROC.search(parse_log_message(ln)))]
    proc_before = [t for (i, t) in proc if i < last_intr]
    s.accepted_turn_before_last_interruption = proc_before[-1] if proc_before else None
    s.response_after_last_interruption = any(i > last_intr for i in resp_idx)


def detect_interruption(s: ConvFinishSignals, base: dict[str, Any]) -> Classification | None:
    """A real accepted turn was cancelled by an interruption and never resumed."""
    if not (
        s.num_interruptions > 0 and s.accepted_turn_before_last_interruption and not s.response_after_last_interruption
    ):
        return None
    return Classification(
        "ended_with_user_interruption",
        {
            **base,
            "accepted_user_turn": s.accepted_turn_before_last_interruption,
            "num_interruptions": s.num_interruptions,
            "responses_after_last_interruption": 0,
        },
    )
