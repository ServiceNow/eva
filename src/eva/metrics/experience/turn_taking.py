"""Turn-taking metric using ElevenLabs audio timestamps (no LLM judge).

Per-turn scores are continuous in [0, 1]. For each turn we use the signal that actually
characterizes what happened on that turn:
  - turn ∈ assistant_interrupted_turns → min(overlap_score, count_score, post_interrupt_score) (capped at 5)
  - turn ∈ user_interrupted_turns      → agent-yield-based score
  - turn ∈ both sets                   → min of the two above
  - otherwise                          → latency-based score

Main turn_taking.score = mean(per-turn scores).

Flat headline sub-metrics (one number each — show up as columns in analysis views):
  Latency:              mean_latency_ms, p50_latency_ms, p90_latency_ms,
                        on_time_rate, early_rate, late_rate
  Agent interruptions:  agent_interruption.rate (always),
                        agent_interruption.num_interruptions (None when rate = 0),
                        agent_interruption.mean_overlap_ms,
                        agent_interruption.mean_overlap_score,
                        agent_interruption.mean_count_score,
                        agent_interruption.mean_post_interrupt_latency_ms,
                        agent_interruption.mean_post_interrupt_latency_score
                        (all except rate and num_interruptions only when rate > 0;
                        the post_interrupt_* pair is additionally gated on at least
                        one interrupt turn having a settled agent response afterward)
  User interruptions:   user_interruption.rate (always),
                        user_interruption.mean_yield_ms,
                        user_interruption.mean_yield_score
                        (the latter two only when rate > 0)

All reported sub-metrics are consistent with the main score: the ``mean_overlap_score``
and ``mean_yield_score`` aggregate exactly the per-turn scores that feed into
``turn_taking.score``. No separate binary "recovered" / "yielded" classifications are
computed (the continuous curves already capture gradation).

Per-turn context (overlap/yield/latency ms, derived scores, and diagnostic
n_interrupt_segments) is preserved in the main metric's details.per_turn_evidence.
"""

import statistics
from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class TurnTakingMetric(CodeMetric):
    """Turn-taking metric derived from ElevenLabs per-turn audio timestamps."""

    name = "turn_taking"
    description = "Turn-taking evaluation based on per-turn latency and interruption behavior"
    category = "experience"

    # --- Latency curve (piecewise linear). 0 outside [LATENCY_HARD_EARLY_MS, LATENCY_HARD_LATE_MS]. ---
    # Ramp up 0 → 1 from LATENCY_HARD_EARLY_MS to LATENCY_SWEET_SPOT_LOW_MS.
    # Flat at 1 from LATENCY_SWEET_SPOT_LOW_MS to LATENCY_SWEET_SPOT_HIGH_MS.
    # Ramp down 1 → 0 from LATENCY_SWEET_SPOT_HIGH_MS to LATENCY_HARD_LATE_MS.
    # LATENCY_HARD_EARLY_MS and LATENCY_SWEET_SPOT_LOW_MS are shared across all turns
    # (tool-call and non-tool-call) because early-response penalties and the ramp-up
    # boundary are unaffected by tool execution time.
    LATENCY_HARD_EARLY_MS: float = -500
    LATENCY_SWEET_SPOT_LOW_MS: float = 500
    LATENCY_SWEET_SPOT_HIGH_MS: float = 2000
    LATENCY_HARD_LATE_MS: float = 5000

    # Tool-call turn variants — more lenient on the upper end since tool execution adds inherent latency.
    LATENCY_SWEET_SPOT_HIGH_MS_TOOL: float = 4000
    LATENCY_HARD_LATE_MS_TOOL: float = 7000

    # --- Agent interruption sub-scores. Each agent-interrupt sub-dimension (overlap, count) is
    # computed directly in [0, AGENT_INTERRUPT_MAX_SCORE] so even a best-case barge-in lands
    # at 0.5 — interrupting the user is never fully "free". ---
    AGENT_INTERRUPT_MAX_SCORE: float = 0.5

    # Overlap duration (ms). Score ramps AGENT_INTERRUPT_MAX_SCORE → 0 over [0, OVERLAP_HARD_MS].
    OVERLAP_HARD_MS: float = 2000

    # Distinct agent-interrupt segments in a single user turn. Score ramps
    # AGENT_INTERRUPT_MAX_SCORE → 0 over [1, INTERRUPT_COUNT_HARD]. Only defined for n ≥ 1.
    INTERRUPT_COUNT_HARD: int = 3  # ≥3 distinct barge-ins → count sub-score = 0

    # --- User interruption (agent yield latency in ms). Score ramps down 1 → 0 over [0, YIELD_HARD_MS]. ---
    YIELD_HARD_MS: float = 2000

    # --- Latency classification thresholds (early / on-time / late rates). ---
    # EARLY_THRESHOLD_MS is shared across all turns — early-response behaviour is not
    # affected by whether a tool call occurred.
    EARLY_THRESHOLD_MS: float = 200  # latency < this ⇒ "early"
    LATE_THRESHOLD_MS: float = 4000  # latency >= this ⇒ "late" (no tool call)
    LATE_THRESHOLD_MS_TOOL: float = 6000  # latency >= this ⇒ "late" (turn with tool call)

    @staticmethod
    def _get_turn_ids_with_turn_taking(context: MetricContext) -> list[int]:
        """Return sorted turn IDs that have both user and assistant audio timestamps (excludes greeting)."""
        return sorted(
            context.audio_timestamps_user_turns.keys() & context.audio_timestamps_assistant_turns.keys() - {0}
        )

    @classmethod
    def _latency_score(cls, latency_ms: float, has_tool_call: bool = False) -> float:
        """Map a single latency (ms) to a score in [0, 1] using the piecewise-linear curve."""
        hard_early = cls.LATENCY_HARD_EARLY_MS
        sweet_low = cls.LATENCY_SWEET_SPOT_LOW_MS
        sweet_high = cls.LATENCY_SWEET_SPOT_HIGH_MS_TOOL if has_tool_call else cls.LATENCY_SWEET_SPOT_HIGH_MS
        hard_late = cls.LATENCY_HARD_LATE_MS_TOOL if has_tool_call else cls.LATENCY_HARD_LATE_MS

        if latency_ms <= hard_early or latency_ms >= hard_late:
            return 0.0
        if latency_ms < sweet_low:
            return (latency_ms - hard_early) / (sweet_low - hard_early)
        if latency_ms <= sweet_high:
            return 1.0
        return (hard_late - latency_ms) / (hard_late - sweet_high)

    @classmethod
    def _overlap_score(cls, overlap_ms: float) -> float:
        """Map agent-interrupt overlap (ms) to a score in [0, AGENT_INTERRUPT_MAX_SCORE]."""
        return max(0.0, cls.AGENT_INTERRUPT_MAX_SCORE * (1.0 - overlap_ms / cls.OVERLAP_HARD_MS))

    @classmethod
    def _yield_score(cls, yield_ms: float) -> float:
        """Map agent yield latency (ms) after a user barge-in to a score in [0, 1]."""
        return max(0.0, 1.0 - yield_ms / cls.YIELD_HARD_MS)

    @classmethod
    def _count_score(cls, n_segments: int) -> float:
        """Map agent-interrupt segment count to a score in [0, AGENT_INTERRUPT_MAX_SCORE].

        n=1 → AGENT_INTERRUPT_MAX_SCORE, ramping linearly to 0 at INTERRUPT_COUNT_HARD.
        Callers must guard n_segments ≥ 1 — count_score is undefined for "no interrupt".
        """
        return max(
            0.0,
            cls.AGENT_INTERRUPT_MAX_SCORE * (1.0 - (n_segments - 1) / (cls.INTERRUPT_COUNT_HARD - 1)),
        )

    @staticmethod
    def _compute_overlap_ms(context: MetricContext, turn_id: int) -> float | None:
        """Total simultaneous-speech duration between user and assistant in this turn (ms).

        Streamed turns have multiple segments interleaved with silence; full-range intersection
        would wildly over-count. Sum pairwise segment intersections instead.
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        a_segs = context.audio_timestamps_assistant_turns.get(turn_id)
        if not u_segs or not a_segs:
            return None
        total_overlap_s = 0.0
        for u_start, u_end in u_segs:
            for a_start, a_end in a_segs:
                total_overlap_s += max(0.0, min(u_end, a_end) - max(u_start, a_start))
        return total_overlap_s * 1000

    @staticmethod
    def _count_agent_interrupt_segments(context: MetricContext, turn_id: int) -> int:
        """Return how many distinct agent audio segments overlap the user's speech in this turn.

        Used to decide whether the agent let the user finish after interrupting once, vs.
        barged in multiple separate times during the same user turn.
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        a_segs = context.audio_timestamps_assistant_turns.get(turn_id)
        if not u_segs or not a_segs:
            return 0
        count = 0
        for a_start, a_end in a_segs:
            for u_start, u_end in u_segs:
                if min(u_end, a_end) - max(u_start, a_start) > 0.001:
                    count += 1
                    break
        return count

    @staticmethod
    def _compute_post_interrupt_latency_ms(context: MetricContext, turn_id: int) -> float | None:
        """Latency from end of user speech to agent's *settled* response (ms).

        An agent-interrupt turn can contain both overlap segments (agent talking during
        the user) and a later "real" response segment after the user finishes. This
        measures the silent gap between ``user_last_end`` and the first agent segment
        starting after it — so a brief barge-in followed by a 10-second wait is penalized.

        Returns None when:
          - there are no audio segments on either side, OR
          - an agent segment *spans* ``user_last_end`` (the agent was already speaking
            continuously through the user's end — there is no silent gap to measure;
            the overlap signal already captures the overtalk), OR
          - no agent segment starts strictly after ``user_last_end`` (e.g., the
            conversation ended before the agent responded again).
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        a_segs = context.audio_timestamps_assistant_turns.get(turn_id)
        if not u_segs or not a_segs:
            return None
        user_last_end = u_segs[-1][1]
        # Agent still speaking at user_last_end → no silent gap; skip the signal.
        if any(a_start <= user_last_end < a_end for a_start, a_end in a_segs):
            return None
        settled_starts = [a_start for a_start, _ in a_segs if a_start > user_last_end]
        if not settled_starts:
            return None
        return (min(settled_starts) - user_last_end) * 1000

    @staticmethod
    def _compute_yield_ms(context: MetricContext, turn_id: int) -> float | None:
        """How long the agent kept speaking after the user barged in at this turn.

        Uses the previous turn's last assistant audio_end (agent's talking tail) minus
        this turn's first user audio_start (barge-in moment).
        """
        u_segs = context.audio_timestamps_user_turns.get(turn_id)
        prev_a_segs = context.audio_timestamps_assistant_turns.get(turn_id - 1)
        if not u_segs or not prev_a_segs:
            return None
        user_barge_in = u_segs[0][0]
        agent_stopped = prev_a_segs[-1][1]
        return max(0.0, agent_stopped - user_barge_in) * 1000

    @classmethod
    def _per_turn_score_and_reason(
        cls,
        context: MetricContext,
        turn_id: int,
        has_tool_call: bool = False,
    ) -> tuple[float, str, dict[str, Any]]:
        """Compute (score, reason, evidence) for a single turn.

        Evidence carries the primary signal (latency_ms / overlap_ms / yield_ms), the derived
        per-turn score, and — for interrupt turns — boolean flags (recovered / yielded) so the
        analysis UI and downstream consumers don't need to recompute them.
        """
        is_agent_int = turn_id in context.assistant_interrupted_turns
        is_user_int = turn_id in context.user_interrupted_turns

        agent_score: float | None = None
        user_score: float | None = None
        evidence: dict[str, Any] = {}

        if is_agent_int:
            overlap_ms = cls._compute_overlap_ms(context, turn_id)
            if overlap_ms is not None:
                agent_score = cls._overlap_score(overlap_ms)
                evidence["overlap_ms"] = round(overlap_ms, 3)
                evidence["overlap_score"] = round(agent_score, 4)
                n_segs = cls._count_agent_interrupt_segments(context, turn_id)
                evidence["n_interrupt_segments"] = n_segs
                if n_segs >= 1:
                    count_score = cls._count_score(n_segs)
                    evidence["interrupt_count_score"] = round(count_score, 4)
                    agent_score = min(agent_score, count_score)
                # Fold in the latency from user-end to the agent's *settled* response (if any),
                # so "interrupted briefly then silent for 10s" is also penalized. Turn score
                # becomes the min of overlap_score, count_score, and the settled-latency score.
                post_ms = cls._compute_post_interrupt_latency_ms(context, turn_id)
                if post_ms is not None:
                    post_score = cls._latency_score(post_ms, has_tool_call=has_tool_call)
                    evidence["post_interrupt_latency_ms"] = round(post_ms, 3)
                    evidence["post_interrupt_latency_score"] = round(post_score, 4)
                    agent_score = min(agent_score, post_score)

        if is_user_int:
            yield_ms = cls._compute_yield_ms(context, turn_id)
            if yield_ms is not None:
                user_score = cls._yield_score(yield_ms)
                evidence["yield_ms"] = round(yield_ms, 3)
                evidence["yield_score"] = round(user_score, 4)

        if agent_score is not None and user_score is not None:
            return min(agent_score, user_score), "dual_interrupt", evidence
        if agent_score is not None:
            return agent_score, "agent_interrupt", evidence
        if user_score is not None:
            return user_score, "user_interrupt", evidence

        latency_s = context.latency_assistant_turns[turn_id]
        latency_ms = latency_s * 1000
        score = cls._latency_score(latency_ms, has_tool_call=has_tool_call)
        evidence["latency_ms"] = round(latency_ms, 3)
        evidence["latency_score"] = round(score, 4)
        return score, "latency", evidence

    @classmethod
    def _build_flat_sub_metrics(
        cls,
        context: MetricContext,
        turn_keys: list[int],
        turns_with_tool_calls: set[int],
        per_turn_evidence: dict[int, dict[str, Any]],
    ) -> dict[str, MetricScore]:
        """Compute the curated flat set of headline sub-metrics.

        Raw per-turn signals (overlap_ms, n_interrupt_segments, post_interrupt_latency_ms,
        yield_ms) are pulled from ``per_turn_evidence`` to avoid recomputing them — they
        were already produced by ``_per_turn_score_and_reason``.

        Rate-style sub-metrics are always emitted (including when the rate is zero — "0% of
        turns had user interruption" is a real signal). Conditional ones (mean_overlap_ms,
        mean_overlap_score, mean_yield_ms, mean_yield_score) are omitted when the underlying
        event set is empty so cross-record aggregates reflect only records with those events.
        """
        total_turns = len(turn_keys)
        if not total_turns:
            return {}

        def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
            return MetricScore(
                name=f"{cls.name}.{key}",
                score=value,
                normalized_score=value if normalized else None,
            )

        # --- Latency ---
        latency_data = [
            (t, context.latency_assistant_turns[t] * 1000) for t in turn_keys if t in context.latency_assistant_turns
        ]
        latencies_ms = [ms for _, ms in latency_data]
        sub: dict[str, MetricScore] = {}
        if latencies_ms:
            sorted_lats = sorted(latencies_ms)
            n = len(sorted_lats)

            def _pct(p: float) -> float:
                return sorted_lats[min(n - 1, int(p * n))]

            early = sum(1 for t, ms in latency_data if ms < cls.EARLY_THRESHOLD_MS)
            late = sum(
                1
                for t, ms in latency_data
                if ms >= (cls.LATE_THRESHOLD_MS_TOOL if t in turns_with_tool_calls else cls.LATE_THRESHOLD_MS)
            )
            on_time = n - early - late

            sub["mean_latency_ms"] = _wrap("mean_latency_ms", round(statistics.mean(latencies_ms), 3), False)
            sub["p50_latency_ms"] = _wrap("p50_latency_ms", round(_pct(0.50), 3), False)
            sub["p90_latency_ms"] = _wrap("p90_latency_ms", round(_pct(0.90), 3), False)
            sub["on_time_rate"] = _wrap("on_time_rate", round(on_time / n, 4), True)
            sub["early_rate"] = _wrap("early_rate", round(early / n, 4), True)
            sub["late_rate"] = _wrap("late_rate", round(late / n, 4), True)

        # --- Agent interruptions (prefixed so readers see the grouping in tables/docs). ---
        agent_turns = [t for t in turn_keys if t in context.assistant_interrupted_turns]
        sub["agent_interruption.rate"] = _wrap(
            "agent_interruption.rate", round(len(agent_turns) / total_turns, 4), True
        )
        overlap_ms_list: list[float] = []
        overlap_scores: list[float] = []
        n_segs_list: list[int] = []
        post_ms_list: list[float] = []
        post_scores: list[float] = []
        for t in agent_turns:
            ev = per_turn_evidence.get(t) or {}
            if "overlap_ms" not in ev:
                continue
            overlap_ms_list.append(ev["overlap_ms"])
            overlap_scores.append(ev["overlap_score"])
            n_segs_list.append(ev.get("n_interrupt_segments", 0))
            if "post_interrupt_latency_ms" in ev:
                post_ms_list.append(ev["post_interrupt_latency_ms"])
                post_scores.append(ev["post_interrupt_latency_score"])
        # num_interruptions: total distinct interrupt segments across all agent-interrupted turns.
        # None when there are no agent interruptions so cross-record aggregates exclude clean runs.
        sub["agent_interruption.num_interruptions"] = MetricScore(
            name=f"{cls.name}.agent_interruption.num_interruptions",
            score=float(sum(n_segs_list)) if n_segs_list else None,
            normalized_score=None,
        )
        if overlap_ms_list:
            sub["agent_interruption.mean_overlap_ms"] = _wrap(
                "agent_interruption.mean_overlap_ms", round(statistics.mean(overlap_ms_list), 3), False
            )
            # mean_overlap_score aggregates the same per-turn scores that feed the main score,
            # so downstream consumers see one number consistent with turn_taking.score itself.
            sub["agent_interruption.mean_overlap_score"] = _wrap(
                "agent_interruption.mean_overlap_score", round(statistics.mean(overlap_scores), 4), True
            )
            count_scores = [cls._count_score(n) for n in n_segs_list if n >= 1]
            if count_scores:
                sub["agent_interruption.mean_count_score"] = _wrap(
                    "agent_interruption.mean_count_score",
                    round(statistics.mean(count_scores), 4),
                    True,
                )
        if post_ms_list:
            sub["agent_interruption.mean_post_interrupt_latency_ms"] = _wrap(
                "agent_interruption.mean_post_interrupt_latency_ms",
                round(statistics.mean(post_ms_list), 3),
                False,
            )
            sub["agent_interruption.mean_post_interrupt_latency_score"] = _wrap(
                "agent_interruption.mean_post_interrupt_latency_score",
                round(statistics.mean(post_scores), 4),
                True,
            )

        # --- User interruptions ---
        user_turns = [t for t in turn_keys if t in context.user_interrupted_turns]
        sub["user_interruption.rate"] = _wrap("user_interruption.rate", round(len(user_turns) / total_turns, 4), True)
        yield_ms_list: list[float] = []
        yield_scores: list[float] = []
        for t in user_turns:
            ev = per_turn_evidence.get(t) or {}
            if "yield_ms" not in ev:
                continue
            yield_ms_list.append(ev["yield_ms"])
            yield_scores.append(ev["yield_score"])
        if yield_ms_list:
            sub["user_interruption.mean_yield_ms"] = _wrap(
                "user_interruption.mean_yield_ms", round(statistics.mean(yield_ms_list), 3), False
            )
            sub["user_interruption.mean_yield_score"] = _wrap(
                "user_interruption.mean_yield_score", round(statistics.mean(yield_scores), 4), True
            )

        return sub

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute turn-taking score and flat sub-metrics."""
        try:
            turn_keys = self._get_turn_ids_with_turn_taking(context)

            turns_with_tool_calls: set[int] = {
                entry["turn_id"] for entry in context.conversation_trace if entry.get("type") == "tool_call"
            }

            per_turn_score: dict[int, float] = {}
            per_turn_reason: dict[int, str] = {}
            per_turn_evidence: dict[int, dict[str, Any]] = {}
            for t in turn_keys:
                _has_tool = t in turns_with_tool_calls
                score, reason, evidence = self._per_turn_score_and_reason(context, t, has_tool_call=_has_tool)
                evidence["has_tool_call"] = _has_tool
                per_turn_score[t] = round(score, 4)
                per_turn_reason[t] = reason
                per_turn_evidence[t] = evidence

            total_turns = max(
                max(context.audio_timestamps_user_turns, default=0),
                max(context.audio_timestamps_assistant_turns, default=0),
            )
            details: dict[str, Any] = {
                "per_turn_score": per_turn_score,
                "per_turn_reason": per_turn_reason,
                "per_turn_evidence": per_turn_evidence,
                "num_turns": total_turns,
                "num_evaluated": len(per_turn_score),
            }

            if not per_turn_score:
                self.logger.info(
                    f"[{context.record_id}] No turns with both user and assistant audio timestamps; skipping metric."
                )
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    details=details,
                    error="No turns with both user and assistant audio timestamps",
                )

            mean_score = statistics.mean(per_turn_score.values())

            return MetricScore(
                name=self.name,
                score=round(mean_score, 4),
                normalized_score=round(mean_score, 4),
                details=details,
                sub_metrics=self._build_flat_sub_metrics(context, turn_keys, turns_with_tool_calls, per_turn_evidence),
            )

        except Exception as e:
            return self._handle_error(e, context)
