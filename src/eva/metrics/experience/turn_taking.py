"""Turn-taking metric using ElevenLabs timestamp data + LLM judge (Gemini)."""

from typing import Any

from eva.metrics.base import MetricContext, TextJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import aggregate_per_turn_scores
from eva.models.results import MetricScore
from eva.utils.json_utils import extract_and_load_json


@register_metric
class TurnTakingMetric(TextJudgeMetric):
    """Turn-taking metric using ElevenLabs timestamps and turns transcript variables.

    Evaluates timing of assistant responses after user utterances using ElevenLabs
    turn-level timestamps and transcript.
    Rating scale: -1 (early/interrupting), 0 (on-time), +1 (late), null (indeterminate)
    Normalized: Lower absolute value is better (0 is perfect).
    """

    name = "turn_taking"
    description = "Turn-taking evaluation"
    category = "experience"
    rating_scale = (-1, 1)
    aggregation = "abs_mean"

    @staticmethod
    def _get_turn_ids_with_turn_taking(context: MetricContext) -> list[int]:
        """Return sorted turn IDs for user-assistant exchange pairs (excludes greeting)."""
        return sorted(context.transcribed_user_turns.keys() & context.transcribed_assistant_turns.keys() - {0})

    def _format_conversation_context(
        self,
        context: MetricContext,
        turn_keys: list[int],
        per_turn_latency: dict[int, float | None],
    ) -> str:
        """Build a unified per-turn context block for the judge prompt.

        Uses pre-computed latencies from _compute_per_turn_latency_and_timing_labels
        to avoid duplicate timestamp lookups.

        Expected = TTS text (what was intended to be spoken).
        Heard = transcript (what STT actually transcribed).
        """
        assistant_interrupted = context.assistant_interrupted_turns
        user_interrupted = context.user_interrupted_turns

        # Reference time (earliest start across all turns) for relative display
        all_timestamps = {**context.audio_timestamps_user_turns, **context.audio_timestamps_assistant_turns}
        all_starts = [segs[0][0] for segs in all_timestamps.values() if segs]
        t0 = min(all_starts) if all_starts else 0

        blocks = []
        for turn_id in turn_keys:
            user_heard = context.transcribed_user_turns.get(turn_id, "")
            asst_heard = context.transcribed_assistant_turns.get(turn_id, "")
            user_expected = context.intended_user_turns.get(turn_id, "")
            asst_expected = context.intended_assistant_turns.get(turn_id, "")

            u_segments = context.audio_timestamps_user_turns.get(turn_id)
            a_segments = context.audio_timestamps_assistant_turns.get(turn_id)
            latency = per_turn_latency.get(turn_id)

            if u_segments and a_segments and latency is not None:
                u_start, u_end = u_segments[0][0] - t0, u_segments[-1][1] - t0
                a_start, a_end = a_segments[0][0] - t0, a_segments[-1][1] - t0

                # Format segment details for granularity (e.g. during interruptions)
                u_seg_str = ", ".join(f"({s - t0:.3f}s-{e - t0:.3f}s)" for s, e in u_segments)
                a_seg_str = ", ".join(f"({s - t0:.3f}s-{e - t0:.3f}s)" for s, e in a_segments)

                # Per-segment latencies: each gap between consecutive user-end and assistant-start
                all_segs = sorted(
                    [("user", s - t0, e - t0) for s, e in u_segments]
                    + [("assistant", s - t0, e - t0) for s, e in a_segments],
                    key=lambda x: x[1],
                )
                seg_latencies = []
                for i in range(len(all_segs) - 1):
                    prev_role, _, prev_end = all_segs[i]
                    next_role, next_start, _ = all_segs[i + 1]
                    if prev_role != next_role:
                        gap = next_start - prev_end
                        seg_latencies.append(f"{prev_role}_end→{next_role}_start: {gap:.3f}s")

                # Interruption annotation
                interruption_label = ""
                if turn_id in user_interrupted:
                    interruption_label = "  ⚠ Based on latency values, the user interrupted the assistant at this turn"
                elif turn_id in assistant_interrupted:
                    interruption_label = "  ⚠ Based on latency values, the assistant interrupted the user at this turn"

                block = (
                    f"Turn {turn_id}:{interruption_label}\n"
                    f"  User: start={u_start:.3f}s  end={u_end:.3f}s  duration={u_end - u_start:.3f}s"
                    f"  segments=[{u_seg_str}]\n"
                    f'    Expected: "{user_expected}"\n'
                    f'    Heard: "{user_heard}"\n'
                    f"  Assistant: start={a_start:.3f}s  end={a_end:.3f}s  duration={a_end - a_start:.3f}s"
                    f"  segments=[{a_seg_str}]\n"
                    f'    Expected: "{asst_expected}"\n'
                    f'    Heard: "{asst_heard}"\n'
                    f"  Segment Transitions: {'; '.join(seg_latencies) if seg_latencies else f'user_end→assistant_start: {latency:.3f}s'}"
                )
            else:
                block = (
                    f"Turn {turn_id}: [Skip consideration due to missing timestamps]\n"
                    f"  User:\n"
                    f'    Expected: "{user_expected}"\n'
                    f'    Heard: "{user_heard}"\n'
                    f"  Assistant:\n"
                    f'    Expected: "{asst_expected}"\n'
                    f'    Heard: "{asst_heard}"'
                )
            blocks.append(block)

        # Add global interruption summary
        header_lines = []
        if assistant_interrupted:
            header_lines.append(f"Turns where assistant interrupted the user: {sorted(assistant_interrupted)}")
        if user_interrupted:
            header_lines.append(f"Turns where user interrupted the assistant: {sorted(user_interrupted)}")

        header = "\n".join(header_lines) + "\n\n" if header_lines else ""
        return header + "\n\n".join(blocks)

    def _compute_per_turn_latency_and_timing_labels(
        self,
        context: MetricContext,
        turn_keys: list[int],
    ) -> tuple[dict[int, float | None], dict[int, str | None]]:
        """Return {turn_id: latency} and {turn_id: timing_label} for each turn key.

        Turns with missing timestamps get None values.

        Latency is user_end -> asst_start in seconds.
        Timing label thresholds:
          latency < 200 ms              -> "Early / Interrupting"
          200 ms <= latency < 4000 ms   -> "On-Time"
          latency >= 4000 ms            -> "Late"
        """
        user_ts = context.audio_timestamps_user_turns
        asst_ts = context.audio_timestamps_assistant_turns
        latencies: dict[int, float | None] = {}
        labels: dict[int, str | None] = {}
        for turn_id in turn_keys:
            u = user_ts.get(turn_id)
            a = asst_ts.get(turn_id)
            if not u or not a:
                if turn_id != len(turn_keys):
                    self.logger.warning(
                        f"[{context.record_id}] Missing audio timestamps at turn {turn_id}/{len(turn_keys)} (user={u}, assistant={a}); skipping turn taking for this turn."
                    )
                latencies[turn_id] = None
                labels[turn_id] = None
                continue
            # Use last user segment end → first assistant segment start
            latency_s = a[0][0] - u[-1][1]
            latencies[turn_id] = round(latency_s, 6)
            latency_ms = latency_s * 1000
            if latency_ms < 200:
                labels[turn_id] = "Early / Interrupting"
            elif latency_ms < 4000:
                labels[turn_id] = "On-Time"
            else:
                labels[turn_id] = "Late"
        return latencies, labels

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute turn-taking score using ElevenLabs data."""
        try:
            turn_keys = self._get_turn_ids_with_turn_taking(context)

            # Identify turn keys where either timestamp is missing.
            _user_ts = context.audio_timestamps_user_turns
            _asst_ts = context.audio_timestamps_assistant_turns
            skipped_turn_ids = {
                turn_id for turn_id in turn_keys if not _user_ts.get(turn_id) or not _asst_ts.get(turn_id)
            }
            turns_missing_timestamps = sorted(skipped_turn_ids)
            if skipped_turn_ids:
                self.logger.info(
                    f"[{context.record_id}] Turns with missing timestamps (will be skipped): {turns_missing_timestamps}"
                )

            per_turn_latency, per_turn_timing_labels = self._compute_per_turn_latency_and_timing_labels(
                context, turn_keys
            )
            conversation_context = self._format_conversation_context(context, turn_keys, per_turn_latency)

            prompt = self.get_judge_prompt(
                conversation_context=conversation_context,
            )
            self.logger.debug(f"Judge prompt (len={len(prompt)}):\n{prompt}")

            messages = [{"role": "user", "content": prompt}]

            response_text = await self.llm_client.generate_text(messages)
            self.logger.debug(f"Judge response (len={len(response_text) if response_text else 0}):\n{response_text}")

            per_turn_judge_timing_ratings = {}  # string labels: "On-Time", "Early / Interrupting", etc.
            numeric_ratings = {}  # numeric [-1, 0, 1, None] for aggregation
            per_turn_judge_timing_explanations = {}
            per_turn_turn_contexts = {}

            if response_text is None:
                self.logger.warning(f"[{context.record_id}] No response from judge, returning empty score.")
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                )

            self.logger.debug(f"Raw judge response: {response_text[:200]}")
            parsed = extract_and_load_json(response_text)
            if not isinstance(parsed, list):
                self.logger.warning(f"[{context.record_id}] Expected list response from judge, got: {type(parsed)}")
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"Unexpected response format: {type(parsed)}",
                )
            # Expected: one rating per [User -> Assistant] pair, greeting excluded.
            # per_turn_timing_labels already reflects this (derived from user_ts keys).
            if len(parsed) != len(turn_keys):
                self.logger.warning(
                    f"[{context.record_id}] Expected {len(turn_keys)} ratings (agent turns, greeting excluded), got {len(parsed)}"
                )
            for response_item in parsed:
                idx = response_item.get("turn_id")
                if idx in skipped_turn_ids:
                    self.logger.info(
                        f"[{context.record_id}] Skipping judge result for turn {idx} (missing timestamps)."
                    )
                    continue
                rating = response_item.get("rating")
                label = response_item.get("label")
                explanation = response_item.get("explanation")
                turn_context = {
                    "user_expected": context.intended_user_turns.get(idx, ""),
                    "assistant_expected": context.intended_assistant_turns.get(idx, ""),
                    "user_heard": context.transcribed_user_turns.get(idx, ""),
                    "assistant_heard": context.transcribed_assistant_turns.get(idx, ""),
                }
                per_turn_judge_timing_ratings[idx] = label
                per_turn_turn_contexts[idx] = turn_context
                if rating not in [-1, 0, 1, None]:
                    self.logger.warning(f"[{context.record_id}] Invalid rating {rating} for turn {idx}")
                    numeric_ratings[idx] = None
                    per_turn_judge_timing_explanations[idx] = f"Invalid rating for turn {idx}: {rating}"
                    continue
                numeric_ratings[idx] = rating

                per_turn_judge_timing_explanations[idx] = explanation

            details: dict[str, Any] = {
                "turns_missing_timestamps": turns_missing_timestamps,
                "per_turn_judge_timing_ratings": per_turn_judge_timing_ratings,
                "per_turn_judge_timing_explanations": per_turn_judge_timing_explanations,
                "per_turn_turn_contexts": per_turn_turn_contexts,
                "per_turn_latency": per_turn_latency,
                "judge_prompt": prompt,
                "judge_raw_response": response_text,
            }

            valid_ratings = [r for r in numeric_ratings if r is not None]
            if not valid_ratings:
                all_missing = set(turn_keys) == skipped_turn_ids
                details["skipped"] = all_missing
                details["skipped_reason"] = "All turns have missing audio timestamps" if all_missing else None
                error = None if all_missing else "All turns failed to evaluate"
                if all_missing:
                    self.logger.info(f"[{context.record_id}] All turns have missing timestamps, skipping metric.")
                else:
                    self.logger.warning(f"[{context.record_id}] All turns failed to evaluate (no valid ratings).")
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None if all_missing else 0.0,
                    error=error,
                    details=details,
                )

            # Compare judge labels against latency-derived labels.
            agreement = sum(
                1
                for k, judge_label in per_turn_judge_timing_ratings.items()
                if per_turn_timing_labels.get(k) is not None
                and judge_label.strip().lower() == per_turn_timing_labels[k].strip().lower()
            )
            n_comparable = sum(1 for k in per_turn_judge_timing_ratings if per_turn_timing_labels.get(k) is not None)
            agreement_ratio = round(agreement / n_comparable, 4) if n_comparable > 0 else 0.0
            self.logger.info(
                f"[{context.record_id}] Label accuracy vs ground truth: {agreement}/{n_comparable} = {agreement_ratio:.3f}"
            )
            details["agreement_with_latency_values"] = agreement_ratio

            aggregated_score = aggregate_per_turn_scores(list(numeric_ratings.values()), self.aggregation)
            if aggregated_score is None:
                self.logger.warning(
                    f"[{context.record_id}] Score aggregation returned None (method={self.aggregation}), returning empty score."
                )
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Aggregation failed",
                    details=details,
                )

            normalized_score = 1.0 - min(1.0, abs(aggregated_score))
            avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0.0

            details["aggregation"] = self.aggregation
            details["num_turns"] = len(turn_keys)
            # Last turn commonly has missing timestamps — count it as not applicable, not a failure
            last_turn = turn_keys[-1] if turn_keys else None
            last_turn_skipped = last_turn is not None and last_turn in skipped_turn_ids
            details["num_not_applicable"] = 1 if last_turn_skipped else 0
            details["num_evaluated"] = len(valid_ratings) + details["num_not_applicable"]

            return MetricScore(
                name=self.name,
                score=round(avg_rating, 3),
                normalized_score=round(normalized_score, 3),
                details=details,
            )

        except Exception as e:
            return self._handle_error(e, context)
