"""Tests for TurnTakingMetric.

Scoring model (continuous, 0–1):
  - Latency → piecewise linear (ramp 0 → 1 over [-500ms, 500ms], flat 1 to 2000ms, ramp 1 → 0 to 5000ms).
  - Agent-interrupt turns → overlap-based, capped at AGENT_INTERRUPT_MAX_SCORE = 0.5.
  - User-interrupt turns → agent yield-latency-based (ramp 1 → 0 over [0, 2000ms]).
  - Both flags on one turn → min of the two.

Sub-metrics (flat, one number each):
  Latency:              mean_latency_ms, p50_latency_ms, p90_latency_ms,
                        on_time_rate, early_rate, late_rate
  Agent interruptions:  agent_interruption.rate (always),
                        agent_interruption.mean_overlap_ms,
                        agent_interruption.mean_overlap_score (only when rate > 0)
  User interruptions:   user_interruption.rate (always),
                        user_interruption.mean_yield_ms,
                        user_interruption.mean_yield_score (only when rate > 0)
"""

import logging

import pytest

from eva.metrics.experience.turn_taking import TurnTakingMetric

from .conftest import make_metric_context


@pytest.fixture
def metric():
    m = TurnTakingMetric()
    m.logger = logging.getLogger("test_turn_taking")
    return m


# ---------- Curve unit tests ----------


class TestLatencyScore:
    @pytest.mark.parametrize(
        "latency_ms, expected",
        [
            (-1000, 0.00),
            (-500, 0.00),
            (-200, 0.30),
            (0, 0.50),
            (200, 0.70),
            (500, 1.00),
            (1000, 1.00),
            (2000, 1.00),
            (2500, 0.8333),
            (3500, 0.5),
            (4500, 0.1667),
            (5000, 0.00),
            (8000, 0.00),
        ],
    )
    def test_latency_score_points(self, metric, latency_ms, expected):
        assert metric._latency_score(latency_ms) == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize(
        "latency_ms, expected",
        [
            # Lower half unchanged: hard_early / sweet_low shared with non-tool curve.
            (-500, 0.00),
            (0, 0.50),
            (500, 1.00),
            # Extended sweet spot [500, 4000] for tool turns.
            (2000, 1.00),
            (3500, 1.00),
            (4000, 1.00),
            # Ramp 1 → 0 over [4000, 7000] (vs [2000, 5000] without a tool call).
            (4500, 0.8333),
            (5500, 0.5),
            (6500, 0.1667),
            (7000, 0.00),
            (9000, 0.00),
        ],
    )
    def test_latency_score_tool_call_curve(self, metric, latency_ms, expected):
        assert metric._latency_score(latency_ms, has_tool_call=True) == pytest.approx(expected, abs=1e-3)


class TestOverlapScore:
    @pytest.mark.parametrize(
        "overlap_ms, expected",
        [
            (0, 0.50),
            (200, 0.45),
            (500, 0.375),
            (1000, 0.25),
            (2000, 0.00),
        ],
    )
    def test_overlap_score(self, metric, overlap_ms, expected):
        assert metric._overlap_score(overlap_ms) == pytest.approx(expected, abs=1e-3)


class TestYieldScore:
    @pytest.mark.parametrize(
        "yield_ms, expected",
        [
            (0, 1.00),
            (200, 0.90),
            (600, 0.70),
            (1000, 0.50),
            (2000, 0.00),
        ],
    )
    def test_yield_score(self, metric, yield_ms, expected):
        assert metric._yield_score(yield_ms) == pytest.approx(expected, abs=1e-3)


class TestCountScore:
    @pytest.mark.parametrize(
        "n_segments, expected",
        [
            # Computed directly in [0, AGENT_INTERRUPT_MAX_SCORE=0.5] — mirrors _overlap_score's cap
            # so any barge-in lands at most at 0.5.
            (1, 0.50),  # single barge-in → cap
            (2, 0.25),  # halfway to hard floor
            (3, 0.00),  # INTERRUPT_COUNT_HARD → saturated
            (4, 0.00),  # saturated beyond hard cap
            (10, 0.00),
        ],
    )
    def test_count_score(self, metric, n_segments, expected):
        assert metric._count_score(n_segments) == pytest.approx(expected, abs=1e-3)


# ---------- End-to-end scenarios ----------


class TestComputeScenarios:
    @pytest.mark.asyncio
    async def test_all_on_time_ideal(self, metric):
        """3 turns all in sweet spot (1s latency) → mean = 1.0."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)], 3: [(10.0, 11.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.5)], 2: [(7.0, 8.5)], 3: [(12.0, 13.5)]},
        )
        result = await metric.compute(context)
        assert result.error is None
        assert result.normalized_score == pytest.approx(1.0, abs=1e-3)
        assert all(r == "latency" for r in result.details["per_turn_reason"].values())

    @pytest.mark.asyncio
    async def test_agent_interrupt_post_interrupt_latency_penalizes_slow_recovery(self, metric):
        """Brief interrupt (overlap_score would be 0.45) but 8s wait for the real response → score 0."""
        context = make_metric_context(
            # User speaks 0–1s. Agent barges in at 0.8 for 200ms overlap, then goes silent until
            # 9s (8s AFTER user end) for its "settled" response — way beyond the latency curve.
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(15.0, 16.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.8, 1.0), (9.0, 10.0)],
                2: [(17.0, 18.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["overlap_ms"] == pytest.approx(200, abs=1)
        assert ev["overlap_score"] == pytest.approx(0.45, abs=1e-3)
        # Post-interrupt latency is 9.0 - 1.0 = 8s, outside the latency curve → score 0.
        assert ev["post_interrupt_latency_ms"] == pytest.approx(8000, abs=1)
        assert ev["post_interrupt_latency_score"] == pytest.approx(0.0, abs=1e-3)
        # Turn score is min of the two signals — even a clean overlap can't save a slow recovery.
        assert result.details["per_turn_score"][1] == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_agent_interrupt_settled_response_on_time(self, metric):
        """Brief interrupt + 1s follow-up → overlap dominates (capped 0.45), post latency score is 1."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.8, 1.0), (2.0, 3.0)],  # overlap 200ms, settled at 2.0 → post = 1000ms → score 1.0
                2: [(7.0, 8.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["post_interrupt_latency_ms"] == pytest.approx(1000, abs=1)
        assert ev["post_interrupt_latency_score"] == pytest.approx(1.0, abs=1e-3)
        # min(0.45, 1.0) = 0.45
        assert result.details["per_turn_score"][1] == pytest.approx(0.45, abs=1e-3)

    @pytest.mark.asyncio
    async def test_agent_interrupt_no_settled_response_omits_post_latency(self, metric):
        """Agent only overlaps user and never emits a later segment → no post_interrupt_latency."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)]},
            audio_timestamps_assistant_turns={1: [(0.5, 1.5)]},  # fully within user speech
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert "post_interrupt_latency_ms" not in ev
        assert "post_interrupt_latency_score" not in ev

    @pytest.mark.asyncio
    async def test_agent_interrupt_continuous_speech_skips_post_latency(self, metric):
        """Agent's interrupt segment spans user_last_end, then keeps streaming — no silent gap.

        Real-world shape: short overlap (170ms) then agent speaks continuously for many seconds,
        split into multiple contiguous streaming chunks. The "post-interrupt latency" should be
        treated as N/A — the agent was already responding, there's no wait to measure.
        """
        context = make_metric_context(
            # User ends at 1.0. Agent segment 1 starts at 0.83 (170ms overlap) and runs to 5.0.
            # Agent segments 2 and 3 are later contiguous chunks of the same ongoing response.
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(10.0, 11.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.83, 5.0), (5.0, 7.5), (7.5, 9.0)],
                2: [(12.0, 13.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        # Overlap detected and scored (still penalized for the barge-in) — no bogus 4s-latency signal.
        assert ev["overlap_ms"] == pytest.approx(170, abs=1)
        assert "post_interrupt_latency_ms" not in ev
        assert "post_interrupt_latency_score" not in ev
        # Turn score reflects only overlap, not a spurious 0 from a fake latency.
        assert result.details["per_turn_score"][1] == pytest.approx(ev["overlap_score"], abs=1e-4)

    @pytest.mark.asyncio
    async def test_agent_interrupt_evidence(self, metric):
        """One agent interrupt segment with 200ms overlap → continuous overlap_score, single barge-in."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(0.8, 2.0)], 2: [(7.0, 8.0)]},  # 200ms overlap at turn 1
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["overlap_ms"] == pytest.approx(200, abs=1)
        assert ev["overlap_score"] == pytest.approx(0.45, abs=1e-3)
        assert ev["n_interrupt_segments"] == 1

    @pytest.mark.asyncio
    async def test_agent_reinterrupts_within_same_turn(self, metric):
        """Multiple agent segments overlap the user's single turn — n_interrupt_segments reflects that."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 10.0)], 2: [(15.0, 16.0)]},
            audio_timestamps_assistant_turns={
                1: [(2.0, 2.5), (5.0, 5.5), (8.0, 8.5)],
                2: [(17.0, 18.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["n_interrupt_segments"] == 3

    @pytest.mark.asyncio
    async def test_user_interrupt_evidence(self, metric):
        """User barges in and agent yields within 100ms → continuous yield_score ≈ 0.95."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(3.0, 4.0)]},
            audio_timestamps_assistant_turns={1: [(2.5, 3.1)], 2: [(5.0, 6.0)]},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][2]
        assert ev["yield_ms"] == pytest.approx(100, abs=1)
        assert ev["yield_score"] == pytest.approx(0.95, abs=1e-3)

    @pytest.mark.asyncio
    async def test_no_timestamps_scores_zero(self, metric):
        """No usable turns means turn-taking failed → score 0, not skipped."""
        context = make_metric_context(
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
        )
        result = await metric.compute(context)
        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert result.error is None
        assert result.skipped is False


# ---------- Sub-metric structure ----------


class TestFlatSubMetrics:
    @pytest.mark.asyncio
    async def test_latency_headlines_populated(self, metric):
        """5 turns, mix of latencies → 6 latency sub-metrics present with expected values."""
        # Latencies: 100ms (early), 300ms, 1000ms, 3000ms, 5000ms (late)
        context = make_metric_context(
            audio_timestamps_user_turns={i: [(i * 10.0, i * 10.0 + 1.0)] for i in range(1, 6)},
            audio_timestamps_assistant_turns={
                1: [(11.1, 12.0)],  # 100ms
                2: [(21.3, 22.0)],  # 300ms
                3: [(32.0, 33.0)],  # 1000ms
                4: [(44.0, 45.0)],  # 3000ms
                5: [(56.0, 57.0)],  # 5000ms
            },
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        for k in ("mean_latency_ms", "p50_latency_ms", "p90_latency_ms", "on_time_rate", "early_rate", "late_rate"):
            assert k in sub
        assert sub["on_time_rate"].score == pytest.approx(0.6)
        assert sub["early_rate"].score == pytest.approx(0.2)
        assert sub["late_rate"].score == pytest.approx(0.2)
        assert sub["p50_latency_ms"].score == pytest.approx(1000, abs=1)
        # Raw-ms sub-metrics are not normalized
        assert sub["mean_latency_ms"].normalized_score is None
        assert sub["p50_latency_ms"].normalized_score is None
        # Rate sub-metrics are normalized
        assert sub["on_time_rate"].normalized_score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_no_agent_interruptions_omits_conditional_subs(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.0)]},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.rate"].score == 0.0
        assert "agent_interruption.mean_overlap_ms" not in sub
        assert "agent_interruption.mean_overlap_score" not in sub
        assert sub["user_interruption.rate"].score == 0.0
        assert "user_interruption.mean_yield_ms" not in sub
        assert "user_interruption.mean_yield_score" not in sub

    @pytest.mark.asyncio
    async def test_agent_interrupt_populates_overlap_sub_metrics(self, metric):
        """Agent-interrupt turn with 200ms overlap → mean_overlap_ms=200, mean_overlap_score=0.45."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(0.8, 2.0)], 2: [(7.0, 8.0)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.rate"].score == pytest.approx(0.5)  # 1 of 2 turns
        assert sub["agent_interruption.mean_overlap_ms"].score == pytest.approx(200, abs=1)
        assert sub["agent_interruption.mean_overlap_score"].score == pytest.approx(0.45, abs=1e-3)
        # mean_overlap_score is normalized (in [0, 1]); raw ms is not.
        assert sub["agent_interruption.mean_overlap_score"].normalized_score == pytest.approx(0.45, abs=1e-3)
        assert sub["agent_interruption.mean_overlap_ms"].normalized_score is None

    @pytest.mark.asyncio
    async def test_post_interrupt_sub_metrics_populated(self, metric):
        """Settled response 1s after user → mean_post_interrupt_latency_* populated."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={
                1: [(0.8, 1.0), (2.0, 3.0)],  # overlap + settled 1000ms later
                2: [(7.0, 8.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.mean_post_interrupt_latency_ms"].score == pytest.approx(1000, abs=1)
        assert sub["agent_interruption.mean_post_interrupt_latency_score"].score == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_post_interrupt_sub_metrics_omitted_when_none(self, metric):
        """No settled response after interrupt → post_interrupt_* sub-metrics omitted."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)]},
            audio_timestamps_assistant_turns={1: [(0.5, 1.5)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert "agent_interruption.mean_post_interrupt_latency_ms" not in sub
        assert "agent_interruption.mean_post_interrupt_latency_score" not in sub

    @pytest.mark.asyncio
    async def test_user_interrupt_populates_yield_sub_metrics(self, metric):
        """User-interrupt turn with 100ms yield → mean_yield_ms=100, mean_yield_score=0.95."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(3.0, 4.0)]},
            audio_timestamps_assistant_turns={1: [(2.5, 3.1)], 2: [(5.0, 6.0)]},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["user_interruption.rate"].score == pytest.approx(0.5)
        assert sub["user_interruption.mean_yield_ms"].score == pytest.approx(100, abs=1)
        assert sub["user_interruption.mean_yield_score"].score == pytest.approx(0.95, abs=1e-3)

    @pytest.mark.asyncio
    async def test_user_interrupt_slow_yield_low_score(self, metric):
        """Agent keeps talking 1500ms after barge-in → mean_yield_score ≈ 0.25."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(3.0, 5.0)]},
            audio_timestamps_assistant_turns={1: [(2.5, 4.5)], 2: [(6.0, 7.0)]},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["user_interruption.mean_yield_ms"].score == pytest.approx(1500, abs=1)
        assert sub["user_interruption.mean_yield_score"].score == pytest.approx(0.25, abs=1e-3)

    @pytest.mark.asyncio
    async def test_agent_interrupt_overlap_uses_pairwise_intersection(self, metric):
        """Multi-segment streamed turn: overlap = sum of pairwise segment intersections.

        Guards against the prior bug where full-range intersection inflated overlap to 20+s.
        """
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 4.0), (18.0, 27.0)]},
            audio_timestamps_assistant_turns={1: [(3.8, 14.0), (30.0, 40.0)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        # Only real simultaneous speech is 3.8–4.0 = 200ms.
        assert result.details["per_turn_evidence"][1]["overlap_ms"] == pytest.approx(200, abs=1)


# ---------- Tool-aware scoring ----------


class TestToolAwareScoring:
    """Turns with tool calls get a more lenient latency curve and late-rate threshold."""

    @pytest.mark.asyncio
    async def test_5s_latency_scores_perfect_on_tool_turn(self, metric):
        """5s wait is 0 on the non-tool curve, but sits inside the tool sweet spot → score 1.0."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(20.0, 21.0)]},
            audio_timestamps_assistant_turns={1: [(6.0, 7.0)], 2: [(22.0, 23.0)]},  # 5s, 1s
            conversation_trace=[
                {"type": "tool_call", "turn_id": 1, "tool_name": "lookup"},
            ],
        )
        result = await metric.compute(context)
        ev1 = result.details["per_turn_evidence"][1]
        ev2 = result.details["per_turn_evidence"][2]
        assert ev1["has_tool_call"] is True
        assert ev2["has_tool_call"] is False
        # Tool turn at 5s latency: inside the extended sweet spot [500ms, 4000ms]? No — 5000ms
        # is on the ramp-down. (7000-5000)/(7000-4000) = 0.6667.
        assert ev1["latency_score"] == pytest.approx(0.6667, abs=1e-3)
        # Non-tool turn at 1s: firmly in the sweet spot.
        assert ev2["latency_score"] == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_tool_turn_beyond_hard_late_scores_zero(self, metric):
        """Latency past 7s on a tool turn still drops to 0 (just further out than the non-tool 5s)."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(20.0, 21.0)]},
            audio_timestamps_assistant_turns={1: [(9.0, 10.0)], 2: [(22.0, 23.0)]},  # 8s, 1s
            conversation_trace=[{"type": "tool_call", "turn_id": 1, "tool_name": "lookup"}],
        )
        result = await metric.compute(context)
        assert result.details["per_turn_evidence"][1]["latency_score"] == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_has_tool_call_false_when_trace_has_no_tool_calls(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.0)]},
            conversation_trace=[
                {"role": "user", "turn_id": 1, "type": "transcribed", "content": "hi"},
                {"role": "assistant", "turn_id": 1, "type": "intended", "content": "hello"},
            ],
        )
        result = await metric.compute(context)
        assert result.details["per_turn_evidence"][1]["has_tool_call"] is False

    @pytest.mark.asyncio
    async def test_late_rate_uses_tool_threshold(self, metric):
        """A 5s latency is 'late' on a non-tool turn but 'on time' on a tool turn (threshold 6s)."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(20.0, 21.0)]},
            audio_timestamps_assistant_turns={1: [(6.0, 7.0)], 2: [(27.0, 28.0)]},  # 5s tool, 6s no-tool
            conversation_trace=[{"type": "tool_call", "turn_id": 1, "tool_name": "lookup"}],
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        # Turn 1 (tool, 5s < LATE_THRESHOLD_MS_TOOL=6000) → on_time.
        # Turn 2 (no tool, 6s >= LATE_THRESHOLD_MS=4000) → late.
        assert sub["late_rate"].score == pytest.approx(0.5, abs=1e-3)
        assert sub["on_time_rate"].score == pytest.approx(0.5, abs=1e-3)

    @pytest.mark.asyncio
    async def test_tool_call_on_turn_id_greater_than_one(self, metric):
        """Tool call on turn 2 (not turn 1) must apply the lenient curve to turn 2, not turn 1.

        Exposes a walrus-operator precedence bug where `turn_id` is assigned the bool result of
        `entry.get("turn_id") and entry.get("type") == "tool_call"` instead of the actual turn ID.
        Because True == 1 in Python, the bug only manifests for turn_id != 1.
        """
        context = make_metric_context(
            # Turn 1: 1s latency, no tool call → non-tool curve → score 1.0 (sweet spot)
            # Turn 2: 5s latency, tool call   → tool curve    → score 0.6667 (ramp-down)
            #                                   non-tool curve → score 0.0   (past hard_late=5000)
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(20.0, 21.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.0)], 2: [(26.0, 27.0)]},  # 1s, 5s
            conversation_trace=[
                {"type": "tool_call", "turn_id": 2, "tool_name": "lookup"},
            ],
        )
        result = await metric.compute(context)
        ev1 = result.details["per_turn_evidence"][1]
        ev2 = result.details["per_turn_evidence"][2]
        assert ev1["has_tool_call"] is False
        assert ev2["has_tool_call"] is True
        # Turn 1: 1s latency, no tool → firmly inside sweet spot.
        assert ev1["latency_score"] == pytest.approx(1.0, abs=1e-3)
        # Turn 2: 5s latency with tool call → on ramp-down (7000-5000)/(7000-4000) = 0.6667.
        # Without the fix this scores 0.0 (non-tool hard_late=5000 treats 5s as beyond the cliff).
        assert ev2["latency_score"] == pytest.approx(0.6667, abs=1e-3)

    @pytest.mark.asyncio
    async def test_post_interrupt_latency_honors_tool_curve(self, metric):
        """Settled response 5s after user_end on a tool turn → non-zero score via the tool curve."""
        context = make_metric_context(
            # Interrupt at 0.8–1.0, then settled response at 6.0 (5s after user end = 1.0).
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(10.0, 11.0)]},
            audio_timestamps_assistant_turns={1: [(0.8, 1.0), (6.0, 7.0)], 2: [(12.0, 13.0)]},
            assistant_interrupted_turns={1},
            conversation_trace=[{"type": "tool_call", "turn_id": 1, "tool_name": "lookup"}],
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        # 5000ms post-interrupt latency on the tool curve: (7000-5000)/(7000-4000) = 0.6667.
        assert ev["post_interrupt_latency_ms"] == pytest.approx(5000, abs=1)
        assert ev["post_interrupt_latency_score"] == pytest.approx(0.6667, abs=1e-3)


# ---------- Count-based interrupt penalty ----------


class TestInterruptCountPenalty:
    @pytest.mark.asyncio
    async def test_single_barge_in_count_score_at_cap(self, metric):
        """n=1 lands at AGENT_INTERRUPT_MAX_SCORE=0.5 (matching the overlap cap)."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 2.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(0.8, 1.0), (3.0, 4.0)], 2: [(7.0, 8.0)]},
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["n_interrupt_segments"] == 1
        assert ev["interrupt_count_score"] == pytest.approx(0.5, abs=1e-3)
        # overlap_score (0.45) < count_score (0.5) → overlap still drives the turn score.
        assert result.details["per_turn_score"][1] == pytest.approx(ev["overlap_score"], abs=1e-4)

    @pytest.mark.asyncio
    async def test_count_score_dominates_multi_barge_in_turn(self, metric):
        """3 tiny overlaps: each overlap_score is near cap, but count_score drops to 0 → turn score 0."""
        context = make_metric_context(
            # Three 50ms agent barge-ins during a single long user turn, then settled response.
            audio_timestamps_user_turns={1: [(0.0, 10.0)], 2: [(15.0, 16.0)]},
            audio_timestamps_assistant_turns={
                1: [(2.0, 2.05), (5.0, 5.05), (8.0, 8.05), (11.0, 12.0)],
                2: [(17.0, 18.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["n_interrupt_segments"] == 3
        assert ev["interrupt_count_score"] == pytest.approx(0.0, abs=1e-3)
        # Overlap is tiny (150ms total) so overlap_score is near cap — count_score=0 forces turn=0.
        assert ev["overlap_score"] > 0.4
        assert result.details["per_turn_score"][1] == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_two_segment_interrupt_halves_cap(self, metric):
        """n=2 is halfway from the cap (0.5) to the floor (0) → 0.25."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 10.0)], 2: [(15.0, 16.0)]},
            audio_timestamps_assistant_turns={
                1: [(2.0, 2.05), (5.0, 5.05), (11.0, 12.0)],
                2: [(17.0, 18.0)],
            },
            assistant_interrupted_turns={1},
        )
        result = await metric.compute(context)
        ev = result.details["per_turn_evidence"][1]
        assert ev["n_interrupt_segments"] == 2
        assert ev["interrupt_count_score"] == pytest.approx(0.25, abs=1e-3)


# ---------- New sub-metrics: num_interruptions, mean_count_score ----------


class TestInterruptCountSubMetrics:
    @pytest.mark.asyncio
    async def test_num_interruptions_sums_segments_across_turns(self, metric):
        """Two interrupt turns (1 seg + 2 segs) → num_interruptions = 3, mean_count_score = mean(1, 0.5)."""
        context = make_metric_context(
            audio_timestamps_user_turns={
                1: [(0.0, 5.0)],
                2: [(10.0, 20.0)],
                3: [(25.0, 26.0)],  # normal latency turn
            },
            audio_timestamps_assistant_turns={
                1: [(1.0, 1.2), (6.0, 7.0)],  # 1 barge-in, then settled
                2: [(11.0, 11.05), (14.0, 14.05), (21.0, 22.0)],  # 2 barge-ins, then settled
                3: [(27.0, 28.0)],
            },
            assistant_interrupted_turns={1, 2},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        assert sub["agent_interruption.num_interruptions"].score == pytest.approx(3.0)
        # num_interruptions is a raw count, not normalized.
        assert sub["agent_interruption.num_interruptions"].normalized_score is None
        # mean_count_score = mean(count_score(1)=0.5, count_score(2)=0.25) = 0.375.
        assert sub["agent_interruption.mean_count_score"].score == pytest.approx(0.375, abs=1e-3)
        assert sub["agent_interruption.mean_count_score"].normalized_score == pytest.approx(0.375, abs=1e-3)

    @pytest.mark.asyncio
    async def test_num_interruptions_none_when_no_agent_interrupts(self, metric):
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(2.0, 3.0)], 2: [(7.0, 8.0)]},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics
        # num_interruptions surfaces None so cross-record aggregates exclude clean runs.
        assert sub["agent_interruption.num_interruptions"].score is None
        assert sub["agent_interruption.num_interruptions"].normalized_score is None
        assert "agent_interruption.mean_count_score" not in sub


# ---------- Dual interrupt ----------


class TestDualInterrupt:
    """Turn present in both assistant_interrupted_turns and user_interrupted_turns.

    Implementation path: both agent_score and user_score are computed; score = min of the two;
    reason = "dual_interrupt"; per-turn evidence carries both overlap_* and yield_* keys.
    Sub-metrics for both interrupt types are populated from the same turn's evidence.
    """

    @pytest.mark.asyncio
    async def test_dual_interrupt_score_is_min_of_agent_and_user(self, metric):
        """Turn 2 is in both interrupt sets → score = min(agent_score, user_score)."""
        context = make_metric_context(
            # Turn 1: user 0–1s, agent 1.5–3.0s (latency 500ms → sweet spot, score 1.0).
            # Turn 2 (dual): user 2.5–5.0s, agent 4.0–6.0s.
            #   Agent interrupt: overlap = max(0, min(5.0,6.0) – max(2.5,4.0)) = 1000ms
            #     → overlap_score = 0.5*(1–1000/2000) = 0.25; n_segs=1 → count_score=0.5
            #     → agent_score = min(0.25, 0.5) = 0.25
            #   User interrupt: yield_ms = max(0, prev_agent_end 3.0 – user_start 2.5)*1000 = 500ms
            #     → yield_score = 1.0 – 500/2000 = 0.75 → user_score = 0.75
            #   dual_score = min(0.25, 0.75) = 0.25
            # Turn 3: clean (latency 1s → score 1.0).
            audio_timestamps_user_turns={
                1: [(0.0, 1.0)],
                2: [(2.5, 5.0)],
                3: [(10.0, 11.0)],
            },
            audio_timestamps_assistant_turns={
                1: [(1.5, 3.0)],
                2: [(4.0, 6.0)],
                3: [(12.0, 13.0)],
            },
            assistant_interrupted_turns={2},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        assert result.error is None

        ev2 = result.details["per_turn_evidence"][2]
        assert result.details["per_turn_reason"][2] == "dual_interrupt"
        assert result.details["per_turn_score"][2] == pytest.approx(0.25, abs=1e-3)

        # Agent-interrupt sub-evidence is populated.
        assert ev2["overlap_ms"] == pytest.approx(1000, abs=1)
        assert ev2["overlap_score"] == pytest.approx(0.25, abs=1e-3)
        assert ev2["n_interrupt_segments"] == 1
        assert ev2["interrupt_count_score"] == pytest.approx(0.5, abs=1e-3)
        # Agent spans user_last_end (4.0 ≤ 5.0 < 6.0) → no settled response gap.
        assert "post_interrupt_latency_ms" not in ev2

        # User-interrupt sub-evidence is populated on the same turn.
        assert ev2["yield_ms"] == pytest.approx(500, abs=1)
        assert ev2["yield_score"] == pytest.approx(0.75, abs=1e-3)

    @pytest.mark.asyncio
    async def test_dual_interrupt_sub_metrics_populated_for_both_interrupt_types(self, metric):
        """agent_interruption.* and user_interruption.* sub-metrics are both emitted."""
        context = make_metric_context(
            audio_timestamps_user_turns={
                1: [(0.0, 1.0)],
                2: [(2.5, 5.0)],
                3: [(10.0, 11.0)],
            },
            audio_timestamps_assistant_turns={
                1: [(1.5, 3.0)],
                2: [(4.0, 6.0)],
                3: [(12.0, 13.0)],
            },
            assistant_interrupted_turns={2},
            user_interrupted_turns={2},
        )
        result = await metric.compute(context)
        sub = result.sub_metrics

        # Agent-interruption sub-metrics: 1 of 3 turns is an agent interrupt.
        assert sub["agent_interruption.rate"].score == pytest.approx(1 / 3, abs=1e-4)
        assert sub["agent_interruption.mean_overlap_ms"].score == pytest.approx(1000, abs=1)
        assert sub["agent_interruption.mean_overlap_score"].score == pytest.approx(0.25, abs=1e-3)

        # User-interruption sub-metrics: 1 of 3 turns is a user interrupt.
        assert sub["user_interruption.rate"].score == pytest.approx(1 / 3, abs=1e-4)
        assert sub["user_interruption.mean_yield_ms"].score == pytest.approx(500, abs=1)
        assert sub["user_interruption.mean_yield_score"].score == pytest.approx(0.75, abs=1e-3)
