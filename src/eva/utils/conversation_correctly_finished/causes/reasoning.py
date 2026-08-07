"""Cause: the model reasoned but produced no spoken answer / tool call (``reasoning_only`` / ``reasoning_too_long``)."""

import csv
from pathlib import Path
from typing import Any

from eva.utils.conversation_correctly_finished.signals import Classification, ConvFinishSignals

csv.field_size_limit(10**7)  # agent_perf_stats rows embed full prompts


def extract_perf_stats(perf: Path, s: ConvFinishSignals) -> None:
    """Reasoning signals from the last agent_perf_stats.csv row."""
    if not perf.exists():
        return
    try:
        rows = list(csv.DictReader(perf.open(newline="")))
    except (OSError, csv.Error):
        s.notes.append("agent_perf_stats unreadable")
        return
    s.num_llm_calls = len(rows)
    if not rows:
        return
    last = rows[-1]
    s.last_perf_response_empty = not (last.get("response") or "").strip()
    s.last_perf_has_tool_call = bool((last.get("tool_calls") or "").strip())
    # The perf-stats writer wraps non-empty reasoning as f'"{content}"' and stores empty reasoning
    # as an empty cell. Strip any wrapping quotes; both cases read as expected. (Older runs may
    # still contain the literal '""' — stripping handles those too.)
    s.last_perf_reasoning = (last.get("reasoning") or "").strip().strip('"').strip()
    try:
        s.last_perf_reasoning_tokens = int(float(last.get("reasoning_tokens") or 0))
    except (TypeError, ValueError):
        s.last_perf_reasoning_tokens = 0
    s.last_perf_stop_reason = (last.get("stop_reason") or "").strip()


def detect_reasoning(s: ConvFinishSignals, base: dict[str, Any]) -> Classification | None:
    """The model reasoned but produced no spoken answer / tool call.

    Reasoning is evidenced by visible text OR a hidden-reasoning token count (gemini/gpt-5 report
    ``reasoning_tokens`` without returning the thinking text). Split by whether it hit the token cap:
    ``stop_reason == 'length'`` → ``reasoning_too_long`` (ran out of tokens mid-reasoning);
    otherwise → ``reasoning_only`` (finished, but only reasoning came out).
    """
    reasoned = s.last_perf_reasoning.strip() or s.last_perf_reasoning_tokens > 0
    if not (s.last_perf_response_empty and not s.last_perf_has_tool_call and reasoned):
        return None
    category = "reasoning_too_long" if s.last_perf_stop_reason == "length" else "reasoning_only"
    return Classification(
        category,
        {
            **base,
            "final_response_empty": True,
            "final_row_had_tool_call": False,
            "reasoning_preview": s.last_perf_reasoning[:300],  # empty when reasoning is hidden
            "reasoning_chars": len(s.last_perf_reasoning),
            "reasoning_tokens": s.last_perf_reasoning_tokens,
            "stop_reason": s.last_perf_stop_reason,
            "num_llm_calls": s.num_llm_calls,
        },
    )
