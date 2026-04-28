"""User plausibility metric for validation."""

import json
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore
from eva.utils.prompt_manager import get_prompt_manager

_USER_BEHAVIORAL_FIDELITY_CORRUPTION_KEYS = (
    "extra_modifications",
    "premature_ending",
    "missing_information",
    "duplicate_modifications",
    "decision_tree_violation",
)

# --- Pipeline-specific prompt text for user behavioral fidelity ---

_CASCADE_CONVERSATION_EVIDENCE = (
    "You are provided with two views of the conversation. Use BOTH when analyzing user behavior.\n\n"
    "### Agent-Side Transcript (includes tool calls)\n"
    "This is the full conversation as seen by the agent, including all tool calls and their results. "
    "IMPORTANT: The user turns in this transcript are the agent's TRANSCRIPTIONS of what the user said "
    "— these may contain transcription errors (e.g., mishearing names, numbers, or codes). Do not "
    "penalize the user for information that was transcribed incorrectly by the agent.\n"
    "{conversation_trace}\n\n"
    "### User-Side Text (ground truth for what the user said)\n"
    "{intended_user_turns}\n"
    "This is what the user actually said out loud during the conversation. When evaluating whether the "
    "user provided correct information, ALWAYS check this source. If there is a discrepancy between the "
    "agent-side transcript and this text, the user-side text is the ground truth — the user said it "
    "correctly and the agent misheard."
)

_AUDIO_NATIVE_CONVERSATION_EVIDENCE = (
    "This is a **speech-to-speech** system — the agent receives raw audio, not a text transcript. "
    "The user turns shown in the conversation trace are the **intended text** (what the user simulator "
    "was instructed to say). Since the agent only hears audio, discrepancies between what the user "
    "intended and how the agent responded may be due to the agent mishearing — do not penalize the user "
    "for the agent's audio perception errors.\n\n"
    "### Conversation (includes tool calls)\n"
    "{conversation_trace}\n\n"
    "### User Intended Text\n"
    "{intended_user_turns}\n"
    "This is what the user simulator was instructed to say. Use this as the ground truth for evaluating "
    "whether the user followed their goal and instructions correctly."
)


@register_metric
class UserBehavioralFidelityMetric(ConversationTextJudgeMetric):
    """User plausibility (corruption) validation metric.

    Evaluates whether the simulated user's behavior unfairly affected the agent
    evaluation — i.e., whether the simulation was "corrupted". Checks for 5
    corruption types: goal drift, premature resolution, information volunteering,
    unnatural persistence, and implausible acceptance.

    Rating scale: 1 (not corrupted), 0 (corrupted)
    Binary rating to identify conversations that need review.
    """

    name = "user_behavioral_fidelity"
    description = "Validation metric for simulated user corruption detection"
    category = "validation"
    rating_scale = (0, 1)
    default_model = "gpt-5.2-medium"

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        modification_tools = [t for t in context.agent_tools if t.get("tool_type") == "write"]
        conversation_evidence_template = (
            _AUDIO_NATIVE_CONVERSATION_EVIDENCE if context.is_audio_native else _CASCADE_CONVERSATION_EVIDENCE
        )
        conversation_evidence = conversation_evidence_template.format(
            conversation_trace=transcript_text,
            intended_user_turns=context.intended_user_turns,
        )

        agent_timeout = is_agent_timeout_on_user_turn(
            context.conversation_ended_reason,
            context.audio_timestamps_user_turns,
            context.audio_timestamps_assistant_turns,
        )
        conversation_end = (
            "the agent's failure to respond to the final user turn."
            if agent_timeout
            else "the user calling the end_call tool."
        )

        return {
            "conversation_evidence": conversation_evidence,
            "modification_tools": json.dumps(modification_tools, indent=2),
            "conversation_end": conversation_end,
            "user_simulator_instructions": _render_user_simulator_instructions(context),
        }

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore with corruption analysis details and per-type detection sub-metrics."""
        corruption_analysis = response.get("corruption_analysis", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=corruption_analysis,
            entry_keys=_USER_BEHAVIORAL_FIDELITY_CORRUPTION_KEYS,
            flag_field="detected",
            detail_fields=("analysis",),
        )

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "corrupted": rating == 0,
                "corruption_analysis": corruption_analysis,
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )


def _render_user_simulator_instructions(context: MetricContext) -> str:
    """Render the user-simulator system prompt for the record's domain.

    Returns the full prompt the user-sim was given, so the judge can evaluate
    user behavior against the user-sim's actual instructions (not just a
    paraphrased rubric).

    Raises:
        ValueError: if user_goal is not a structured dict.
    """
    if not isinstance(context.user_goal, dict):
        raise ValueError(
            "user_behavioral_fidelity requires context.user_goal to be a dict "
            f"(with 'decision_tree', 'high_level_user_goal', etc.); got "
            f"{type(context.user_goal).__name__}."
        )

    domain = context.agent_id.removeprefix("agent_")
    decision_tree = context.user_goal.get("decision_tree", {})
    return get_prompt_manager().get_prompt(
        f"user_simulator.system_prompt_{domain}",
        high_level_user_goal=context.user_goal.get("high_level_user_goal", ""),
        must_have_criteria=decision_tree.get("must_have_criteria", []),
        nice_to_have_criteria=decision_tree.get("nice_to_have_criteria", []),
        negotiation_behavior=decision_tree.get("negotiation_behavior", []),
        resolution_condition=decision_tree.get("resolution_condition", ""),
        failure_condition=decision_tree.get("failure_condition", ""),
        escalation_behavior=decision_tree.get("escalation_behavior", ""),
        edge_cases=decision_tree.get("edge_cases", []),
        information_required=context.user_goal.get("information_required", {}),
        user_persona=context.user_persona,
        starting_utterance=context.user_goal.get("starting_utterance", ""),
        current_date_time=context.current_date_time,
    )
