"""Agent speech fidelity metric for S2S models — entity-focused evaluation.

For S2S (speech-to-speech) models, there is no intended text to compare against.
Instead, this metric verifies that key entities spoken by the agent (from tool
responses and user utterances) are accurate by sending a redacted conversation
trace alongside the agent audio to Gemini.
"""

import json
from typing import Any

from eva.metrics.base import MetricContext
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric
from eva.metrics.utils import aggregate_per_turn_scores, normalize_rating, resolve_turn_id
from eva.models.results import MetricScore


class AgentSpeechFidelityS2SMetric(SpeechFidelityBaseMetric):
    """Audio-based entity fidelity metric for S2S agent speech.

    Evaluates whether key entities (from tool responses and user utterances) are
    spoken correctly by the agent, without requiring intended text.

    Rating scale: 0 (entity error) or 1 (all entities accurate)
    """

    name = "agent_speech_fidelity"
    description = "Audio-based evaluation of agent entity fidelity for S2S models"
    category = "accuracy"
    role = "assistant"
    rating_scale = (0, 1)
    pass_at_k_threshold = 0.95

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute entity fidelity score using redacted conversation trace + audio."""
        try:
            audio_segment = self.load_role_audio(context, self.role)
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"No {self.role} audio file available",
                )

            audio_segment = self._trim_silence(audio_segment, context)

            redacted_trace = self._build_redacted_trace(context)
            assistant_turn_ids = self._get_assistant_turn_ids(redacted_trace)

            if not assistant_turn_ids:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No assistant turns found in conversation trace",
                )

            num_turns = len(assistant_turn_ids)
            trace_formatted = self._format_redacted_trace(redacted_trace)
            audio_b64 = self.encode_audio_segment(audio_segment)

            prompt = self.get_judge_prompt(
                prompt_key="s2s_user_prompt",
                conversation_trace_formatted=trace_formatted,
            )

            messages = self.create_audio_message(audio_b64, prompt)

            per_turn_ratings: dict[int, int | None] = {}
            per_turn_explanations: dict[int, str] = {}
            per_turn_transcripts: dict[int, str] = {}
            per_turn_normalized: dict[int, float] = {}
            min_rating, max_rating = self.rating_scale
            valid_ratings_range = list(range(min_rating, max_rating + 1))

            response_text, turns = await self._call_and_parse(messages, context, audio_segment, prompt)

            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                )

            self.logger.debug(f"Raw judge response: {response_text[:200]}")

            if len(turns) != num_turns:
                self.logger.warning(
                    f"[{context.record_id}] Expected {num_turns} ratings for S2S entity fidelity, got {len(turns)}"
                )

            per_turn_has_entities: dict[int, bool] = {}

            for response_item in turns:
                turn_id = resolve_turn_id(response_item, assistant_turn_ids, self.name)
                if turn_id is None:
                    continue
                rating = response_item.get("rating")
                transcript = response_item.get("transcript")
                explanation = response_item.get("explanation", "")
                has_entities = response_item.get("has_entities", True)

                per_turn_has_entities[turn_id] = has_entities

                if not has_entities:
                    # Exclude turns with no entities from scoring
                    per_turn_ratings[turn_id] = rating
                    per_turn_explanations[turn_id] = explanation
                    per_turn_transcripts[turn_id] = transcript
                    continue

                if rating not in valid_ratings_range:
                    self.logger.warning(f"[{context.record_id}] Invalid rating {rating} for turn {turn_id}")
                    per_turn_ratings[turn_id] = None
                    per_turn_explanations[turn_id] = f"Invalid rating: {rating}"
                    continue

                per_turn_ratings[turn_id] = rating
                per_turn_explanations[turn_id] = explanation
                per_turn_transcripts[turn_id] = transcript
                per_turn_normalized[turn_id] = normalize_rating(rating, min_rating, max_rating)

            aggregated_score = aggregate_per_turn_scores(list(per_turn_normalized.values()), self.aggregation)

            # Only count turns with entities toward the score
            valid_ratings = [
                per_turn_ratings[tid]
                for tid in per_turn_ratings
                if per_turn_ratings[tid] is not None and per_turn_has_entities.get(tid, True)
            ]
            num_skipped_no_entities = sum(1 for v in per_turn_has_entities.values() if not v)

            # No valid scores to aggregate — not an error, just nothing to score
            skipped = not valid_ratings
            avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None

            details: dict[str, Any] = {
                "variant": "s2s",
                "aggregation": self.aggregation,
                "num_turns": num_turns,
                "num_evaluated": len(valid_ratings),
                "num_skipped_no_entities": num_skipped_no_entities,
                "skipped_reason": "No valid ratings to aggregate" if skipped else None,
                "per_turn_ratings": per_turn_ratings,
                "per_turn_has_entities": per_turn_has_entities,
                "per_turn_explanations": per_turn_explanations,
                "judge_prompt": prompt,
                "judge_raw_response": response_text,
            }

            return MetricScore(
                name=self.name,
                score=round(avg_rating, 3) if avg_rating is not None else None,
                normalized_score=round(aggregated_score, 3) if aggregated_score is not None else None,
                details=details,
                skipped=skipped,
            )

        except Exception as e:
            return self._handle_error(e, context)

    @staticmethod
    def _build_redacted_trace(context: MetricContext) -> list[dict]:
        """Build a redacted conversation trace for entity fidelity evaluation.

        Keeps user entries and tool responses as-is (entity sources).
        Replaces assistant entries with a single placeholder per turn_id
        (a turn can have multiple assistant entries, e.g. before/after tool calls).
        Drops tool_call entries (parameters, not entity sources).

        Note: conversation trace entries use different schemas by type:
        - user/assistant entries have ``role`` + ``content``
        - tool entries have ``type`` (tool_call/tool_response) + ``tool_name`` + data fields
        """
        redacted = []
        seen_assistant_turns: set[int] = set()
        for entry in context.conversation_trace or []:
            role = entry.get("role")
            entry_type = entry.get("type")

            if role == "assistant":
                turn_id = entry.get("turn_id")
                if turn_id not in seen_assistant_turns:
                    seen_assistant_turns.add(turn_id)
                    redacted.append(
                        {
                            "role": "assistant",
                            "turn_id": turn_id,
                            "redacted": True,
                        }
                    )
            elif role == "user":
                redacted.append(
                    {
                        "role": "user",
                        "content": entry.get("content", ""),
                        "turn_id": entry.get("turn_id"),
                    }
                )
            elif entry_type == "tool_response":
                redacted.append(
                    {
                        "role": "tool_response",
                        "tool_name": entry.get("tool_name", "unknown"),
                        "content": entry.get("tool_response", {}),
                        "turn_id": entry.get("turn_id"),
                    }
                )
            # Skip tool_call entries — parameters are not entity sources

        return redacted

    @staticmethod
    def _get_assistant_turn_ids(redacted_trace: list[dict]) -> list[int]:
        """Extract sorted unique assistant turn IDs from the redacted trace."""
        turn_ids = set()
        for entry in redacted_trace:
            if entry.get("role") == "assistant" and entry.get("turn_id") is not None:
                turn_ids.add(entry["turn_id"])
        return sorted(turn_ids)

    @staticmethod
    def _format_redacted_trace(redacted_trace: list[dict]) -> str:
        """Format the redacted trace as text for the prompt."""
        lines = []
        for entry in redacted_trace:
            turn_id = entry.get("turn_id", "?")
            role = entry["role"]

            if role == "user":
                lines.append(f"Turn {turn_id} - User: {entry['content']}")
            elif role == "assistant":
                lines.append(f"Turn {turn_id} - [Assistant speaks]")
            elif role == "tool_response":
                tool_name = entry.get("tool_name", "unknown")
                content = entry.get("content", {})
                if isinstance(content, (dict, list)):
                    content_str = json.dumps(content, indent=None)
                else:
                    content_str = str(content)
                lines.append(f"Turn {turn_id} - Tool Response ({tool_name}): {content_str}")

        return "\n".join(lines)
