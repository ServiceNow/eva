"""Text-only test flow for EVA.

Runs records through the agent/user-simulator loop using pure text
(LLM calls only — no audio, no WebSocket, no TTS/STT). Saves a debug trace
and computes text-compatible metrics.

Usage:
    # Run all records in the dataset
    EVA_DOMAIN=airline LLM_MODEL=gpt-5.2 \
        python scripts/run_text_only.py

    # Run a single record
    EVA_DOMAIN=airline LLM_MODEL=gpt-5.2 \
        python scripts/run_text_only.py --record-id 1.1.2

    # Run with specific metrics
    python scripts/run_text_only.py \
        --metrics task_completion,faithfulness_judge
"""

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.agentic.system import AgenticSystem
from eva.assistant.services.llm import LiteLLMClient
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.metrics.aggregation import (
    EVACompositeDefinition,
    compute_record_aggregates,
    compute_run_level_aggregates,
)
from eva.metrics.base import MetricContext
from eva.metrics.registry import get_global_registry
from eva.metrics.runner import MetricsRunner
from eva.models.agents import AgentConfig
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, MetricScore, RecordMetrics
from eva.utils import router
from eva.utils.hash_utils import get_dict_hash
from eva.utils.log_processing import (
    extract_tool_params_and_responses,
    get_entry_for_audit_log,
    group_consecutive_turns,
)
from eva.utils.logging import get_logger, setup_logging
from eva.utils.prompt_manager import PromptManager

load_dotenv()
setup_logging(level="INFO")

logger = get_logger(__name__)

# Text-compatible metrics that can be run without audio
TEXT_COMPATIBLE_METRICS = [
    "task_completion",
    "tool_call_validity",
    "authentication_success",
    "faithfulness",
    "conversation_progression",
    "conciseness",
    "speakability",
    "user_behavioral_fidelity",
]

# Default text-only EVA composite definitions (no audio metrics).
# Override via --accuracy-metrics and --experience-metrics CLI args.
DEFAULT_ACCURACY_METRICS = ["task_completion", "faithfulness"]
DEFAULT_ACCURACY_THRESHOLDS = {
    "task_completion": ("==", 1.0),
    "faithfulness": (">=", 0.5),
}
DEFAULT_EXPERIENCE_METRICS = ["conversation_progression", "conciseness", "text_response_latency"]
DEFAULT_EXPERIENCE_THRESHOLDS = {
    "conversation_progression": (">=", 0.5),
    "conciseness": (">=", 0.5),
    "text_response_latency": (">=", 0.5),
}


# Latency thresholds in seconds
_IDEAL_LATENCY = 1.0
_MAX_LATENCY = 3.0


def build_text_composites(
    accuracy_metrics: list[str],
    accuracy_thresholds: dict[str, tuple[str, float]],
    experience_metrics: list[str],
    experience_thresholds: dict[str, tuple[str, float]],
) -> list[EVACompositeDefinition]:
    """Build EVA composite definitions for text-only evaluation."""
    all_metrics = accuracy_metrics + experience_metrics
    return [
        EVACompositeDefinition(
            name="EVA-A_pass",
            component_metrics=accuracy_metrics,
            aggregation_type="pass",
            thresholds=accuracy_thresholds,
        ),
        EVACompositeDefinition(
            name="EVA-X_pass",
            component_metrics=experience_metrics,
            aggregation_type="pass",
            thresholds=experience_thresholds,
        ),
        EVACompositeDefinition(
            name="EVA-A_mean",
            component_metrics=accuracy_metrics,
            aggregation_type="mean",
        ),
        EVACompositeDefinition(
            name="EVA-X_mean",
            component_metrics=experience_metrics,
            aggregation_type="mean",
        ),
        EVACompositeDefinition(
            name="EVA-overall_mean",
            component_metrics=all_metrics,
            aggregation_type="mean",
        ),
        EVACompositeDefinition(
            name="EVA-overall_pass",
            component_metrics=[],
            aggregation_type="derived",
            derived_from=["EVA-A_pass", "EVA-X_pass"],
        ),
    ]


END_CALL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "Call this when the conversation is complete and you want to hang up.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_paths(domain: str) -> tuple[Path, Path, Path]:
    """Resolve dataset, scenario-db-dir, and agent-config paths from a domain name."""
    dataset = Path(f"data/{domain}_dataset.jsonl")
    scenario_db_dir = Path(f"data/{domain}_scenarios")
    agent_config = Path(f"configs/agents/{domain}_agent.yaml")

    for label, p in [("dataset", dataset), ("scenario-db-dir", scenario_db_dir), ("agent-config", agent_config)]:
        if not p.exists():
            sys.exit(f"Error: {label} path does not exist: {p}")

    return dataset, scenario_db_dir, agent_config


def build_user_sim_prompt(record: EvaluationRecord, user_simulator_context: str) -> str:
    """Build the user-simulator system prompt from the record's goal and persona."""
    pm = PromptManager()
    goal = record.user_goal
    return pm.get_prompt(
        "user_simulator.system_prompt",
        user_simulator_context=user_simulator_context,
        high_level_user_goal=goal["high_level_user_goal"],
        must_have_criteria=goal["decision_tree"]["must_have_criteria"],
        escalation_behavior=goal["decision_tree"]["escalation_behavior"],
        nice_to_have_criteria=goal["decision_tree"]["nice_to_have_criteria"],
        negotiation_behavior=goal["decision_tree"]["negotiation_behavior"],
        resolution_condition=goal["decision_tree"]["resolution_condition"],
        failure_condition=goal["decision_tree"]["failure_condition"],
        edge_cases=goal["decision_tree"]["edge_cases"],
        information_required=goal["information_required"],
        user_persona=record.user_config["user_persona"],
        starting_utterance=goal["starting_utterance"],
        current_date_time=record.current_date_time,
    )


def build_user_sim_history(audit_log: AuditLog) -> list[dict[str, str]]:
    """Build a simple alternating user/assistant message history for the user-simulator LLM."""
    history: list[dict[str, str]] = []
    for entry in audit_log.transcript:
        msg_type = entry.get("message_type")
        if msg_type == "user":
            history.append({"role": "assistant", "content": entry["value"]})  # from user-sim's POV, user = assistant
        elif msg_type == "assistant":
            history.append({"role": "user", "content": entry["value"]})  # agent responses appear as "user" to the sim
    return history


async def run_user_turn(
    llm_client: LiteLLMClient,
    user_system_prompt: str,
    conversation_history: list[dict[str, str]],
) -> tuple[str, bool]:
    """Generate one user-simulator turn. Returns (message, should_end)."""
    messages = [{"role": "system", "content": user_system_prompt}] + conversation_history
    response, _ = await llm_client.complete(messages, tools=END_CALL_TOOL)

    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls and any(tc.function.name == "end_call" for tc in tool_calls):
        # User simulator decided to end the call
        content = getattr(response, "content", "") or ""
        return content, True

    content = getattr(response, "content", "") or (response if isinstance(response, str) else "")

    # Fallback: detect end_call emitted as text instead of structured tool call.
    if "functions.end_call" in content or ("end_call" in content and content.strip().startswith(("{", "[", "<"))):
        return "", True

    return content, False


# ---------------------------------------------------------------------------
# Context building for metrics (no pipecat/elevenlabs logs needed)
# ---------------------------------------------------------------------------


def build_conversation_trace(audit_log_data: dict) -> list[dict]:
    """Build conversation_trace from audit log, matching the processor's format.

    Normalises raw audit_log transcript entries into the same event format used
    by ``MetricsContextProcessor`` and then converts them via
    ``get_entry_for_audit_log`` so the resulting trace entries are identical to
    what the voice pipeline produces (minus pipecat/elevenlabs enrichments).
    """
    transcript = audit_log_data.get("transcript", [])

    # Step 1: normalise to the processor's unified event format
    events = []
    for entry in transcript:
        event_type = entry.get("message_type", "unknown")
        if event_type == "llm_call":
            continue
        events.append(
            {
                "timestamp_ms": int(entry.get("timestamp", 0)),
                "source": "audit_log",
                "event_type": event_type,
                "data": entry.get("value", {}),
            }
        )

    # Step 2: convert to trace entries using get_entry_for_audit_log,
    # advancing turn_id on each user message (same as the processor).
    turn_id = 1  # Start at 1: turn 0 is the assistant greeting (absent in text-only)
    assistant_spoke = False
    entries: list[dict] = []
    for event in events:
        if event["event_type"] == "user":
            if assistant_spoke:
                turn_id += 1
                assistant_spoke = False
            entries.append(get_entry_for_audit_log(event, turn_id))
        elif event["event_type"] == "assistant":
            assistant_spoke = True
            entry = get_entry_for_audit_log(event, turn_id)
            entries.append(entry)
        elif event["event_type"] in ("tool_call", "tool_response"):
            entries.append(get_entry_for_audit_log(event, turn_id))

    return group_consecutive_turns(entries)


def build_per_role_turns(turns_with_tools: list[dict]) -> tuple[dict[int, str], dict[int, str]]:
    """Extract per-role turn dicts from grouped transcript. Returns (assistant_turns, user_turns)."""
    assistant_turns: dict[int, str] = {}
    user_turns: dict[int, str] = {}
    asst_idx = 0
    user_idx = 0
    for entry in turns_with_tools:
        role = entry.get("role")
        if role == "assistant":
            assistant_turns[asst_idx] = entry.get("content", "")
            asst_idx += 1
        elif role == "user":
            user_turns[user_idx] = entry.get("content", "")
            user_idx += 1
    return assistant_turns, user_turns


def _build_per_turn_calls(raw_transcript: list[dict]) -> dict[int, dict]:
    """Build per-turn data from the raw audit log transcript.

    Returns a dict mapping turn_id -> {"latency_s": float, "calls": [...]}, e.g.:
        {"latency_s": 7.24, "calls": [{"type": "llm", "latency_s": 2.3}, {"type": "tool", "name": "rebook_flight", "latency_s": 0.0}, ...]}

    Turn IDs follow the same convention as build_conversation_trace: first user message is
    turn 1, increments on each user message that follows an assistant response.
    """
    result: dict[int, dict] = {}
    turn_id = 1
    assistant_spoke = False
    in_turn = False
    user_ts: int | None = None
    prev_ts: int | None = None
    calls: list[dict] = []
    pending_tool_name: str | None = None
    pending_tool_ts: int | None = None

    for entry in raw_transcript:
        mt = entry.get("message_type")
        ts = int(entry.get("timestamp", 0))
        v = entry.get("value", {})

        if mt == "user":
            if assistant_spoke:
                turn_id += 1
                assistant_spoke = False
            in_turn = True
            user_ts = ts
            prev_ts = ts
            calls = []
            pending_tool_name = None
            pending_tool_ts = None
        elif mt == "llm_call" and in_turn and prev_ts is not None:
            calls.append({"type": "llm", "latency_s": round((ts - prev_ts) / 1000.0, 3)})
            prev_ts = ts
        elif mt == "tool_call" and in_turn:
            pending_tool_name = v.get("tool", "unknown")
            pending_tool_ts = ts
        elif mt == "tool_response" and in_turn and pending_tool_ts is not None:
            calls.append(
                {"type": "tool", "name": pending_tool_name, "latency_s": round((ts - pending_tool_ts) / 1000.0, 3)}
            )
            prev_ts = ts
            pending_tool_name = None
            pending_tool_ts = None
        elif mt == "assistant":
            if in_turn and user_ts is not None:
                latency_s = round((ts - user_ts) / 1000.0, 3)
                if 0 < latency_s < 1000:
                    result[turn_id] = {"latency_s": latency_s, "calls": list(calls)}
                in_turn = False
            assistant_spoke = True

    return result


def compute_response_latency(raw_transcript: list[dict]) -> MetricScore:
    """Compute per-turn response latency from the raw audit log transcript.

    Measures wall-clock time from a user message to the completed assistant
    response (including all intermediate tool calls), with a per-call breakdown.

    Scoring per turn:
        <= 1s  -> 1.0
        1-3s   -> linear interpolation from 1.0 to 0.0
        > 3s   -> 0.0
    """
    per_turn_data = _build_per_turn_calls(raw_transcript)

    per_turn: list[dict] = []
    for turn_id, data in per_turn_data.items():
        latency_s = data["latency_s"]
        norm = (
            1.0
            if latency_s <= _IDEAL_LATENCY
            else max(0.0, 1.0 - (latency_s - _IDEAL_LATENCY) / (_MAX_LATENCY - _IDEAL_LATENCY))
        )
        per_turn.append(
            {
                "turn_id": turn_id,
                "latency_s": latency_s,
                "normalized": round(norm, 4),
                "calls": data["calls"],
            }
        )

    if not per_turn:
        return MetricScore(
            name="text_response_latency",
            score=0.0,
            normalized_score=None,
            error="No user->assistant turn pairs with timestamps found",
        )

    latencies = [t["latency_s"] for t in per_turn]
    norm_scores = [t["normalized"] for t in per_turn]
    mean_latency = round(sum(latencies) / len(latencies), 3)
    mean_norm = round(sum(norm_scores) / len(norm_scores), 4)

    return MetricScore(
        name="text_response_latency",
        score=mean_latency,
        normalized_score=mean_norm,
        details={
            "mean_latency_s": mean_latency,
            "max_latency_s": round(max(latencies), 3),
            "min_latency_s": round(min(latencies), 3),
            "num_turns": len(per_turn),
            "per_turn": per_turn,
        },
    )


# ---------------------------------------------------------------------------
# Pretty-print trace
# ---------------------------------------------------------------------------


def write_trace(
    output_dir: Path, record: EvaluationRecord, audit_log_data: dict, end_reason: str, turn_count: int, max_turns: int
) -> None:
    """Write a human-readable trace.txt."""
    lines = [
        f"Record: {record.id}",
        f"Category: {record.category or 'N/A'}",
        f"Expected flow: {record.expected_flow}",
        f"User goal: {record.user_goal.get('high_level_user_goal', '')}",
        "=" * 60,
        "",
    ]

    for entry in audit_log_data.get("transcript", []):
        msg_type = entry.get("message_type")
        value = entry.get("value", "")

        if msg_type == "user":
            lines.append(f"[USER] {value}")
        elif msg_type == "assistant":
            lines.append(f"[AGENT] {value}")
        elif msg_type == "tool_call" and isinstance(value, dict):
            params_str = json.dumps(value.get("parameters", {}), indent=None)
            lines.append(f"  [TOOL] {value.get('tool')}({params_str})")
        elif msg_type == "tool_response" and isinstance(value, dict):
            resp = value.get("response") or value.get("error", "")
            resp_str = json.dumps(resp, indent=None) if isinstance(resp, dict) else str(resp)
            # Truncate long tool responses
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "..."
            lines.append(f"  [RESULT] {resp_str}")
        elif msg_type == "llm_call":
            pass  # Skip LLM call entries in trace

    lines.append("")
    lines.append(f"--- Conversation ended: {end_reason} (turn {turn_count}/{max_turns}) ---")

    (output_dir / "trace.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_record(
    record: EvaluationRecord,
    agent: AgentConfig,
    raw_agent_config: dict,
    tool_module: str,
    agent_config_path: Path,
    scenario_db_dir: Path,
    llm_model: str,
    output_dir: Path,
    max_turns: int,
    requested_metrics: list[str],
    output_id: str | None = None,
    composites: list[EVACompositeDefinition] | None = None,
) -> dict[str, Any]:
    """Run a single record through the text-only conversation loop and compute metrics."""
    output_id = output_id or record.id
    record_output_dir = output_dir / output_id
    record_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Init components ----
    scenario_db_path = str(scenario_db_dir / f"{record.id}.json")

    tool_executor = ToolExecutor(
        tool_config_path=str(agent_config_path),
        scenario_db_path=scenario_db_path,
        tool_module_path=tool_module,
        current_date_time=record.current_date_time,
    )
    initial_db = copy.deepcopy(tool_executor.db)
    initial_hash = get_dict_hash(initial_db)

    audit_log = AuditLog()
    agent_llm_client = LiteLLMClient(model=llm_model)
    user_sim_llm_client = LiteLLMClient(model="gpt-5.2")

    agentic_system = AgenticSystem(
        current_date_time=record.current_date_time,
        agent=agent,
        tool_handler=tool_executor,
        audit_log=audit_log,
        llm_client=agent_llm_client,
        output_dir=record_output_dir,
    )

    user_prompt = build_user_sim_prompt(record, user_simulator_context=agent.user_simulator_context)

    # ---- Conversation loop ----
    logger.info(f"Text-only test: record {record.id} | model={llm_model} | max_turns={max_turns}")
    logger.info(f"Metrics: {', '.join(requested_metrics)}")

    user_message = record.user_goal["starting_utterance"]
    end_reason = "max_turns"
    turn_count = 0
    started_at = datetime.now(timezone.utc)

    for turn in range(max_turns):
        turn_count = turn + 1
        logger.info(f"--- Turn {turn_count} ---")
        logger.info(f"  [USER] {user_message}")

        # Agent turn
        agent_response_parts: list[str] = []
        async for part in agentic_system.process_query(user_message):
            agent_response_parts.append(part)
        agent_response = " ".join(agent_response_parts)
        logger.info(f"  [AGENT] {agent_response}")

        # Check for transfer
        if "Transferring you to a live agent" in agent_response:
            end_reason = "transfer"
            break

        # User-simulator turn
        user_history = build_user_sim_history(audit_log)
        user_message, should_end = await run_user_turn(user_sim_llm_client, user_prompt, user_history)
        if should_end:
            end_reason = "goodbye"
            if user_message:
                logger.info(f"  [USER] {user_message} (ending call)")
            else:
                logger.info("  [USER] (ended call)")
            break
    else:
        end_reason = "max_turns"

    ended_at = datetime.now(timezone.utc)
    duration = (ended_at - started_at).total_seconds()
    logger.info(f"Conversation ended: {end_reason} ({turn_count} turns, {duration:.1f}s)")

    # ---- Save outputs ----
    audit_log.save(record_output_dir / "audit_log.json")

    final_db = tool_executor.db
    final_hash = get_dict_hash(final_db)

    with open(record_output_dir / "initial_scenario_db.json", "w") as f:
        json.dump(initial_db, f, indent=2)
    with open(record_output_dir / "final_scenario_db.json", "w") as f:
        json.dump(final_db, f, indent=2)

    stats = audit_log.get_stats()
    conv_result = ConversationResult(
        record_id=record.id,
        completed=True,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration,
        output_dir=str(record_output_dir),
        audit_log_path=str(record_output_dir / "audit_log.json"),
        num_turns=stats["num_turns"],
        num_tool_calls=stats["num_tool_calls"],
        tools_called=stats["tools_called"],
        conversation_ended_reason=end_reason,
        initial_scenario_db_hash=initial_hash,
        final_scenario_db_hash=final_hash,
    )
    with open(record_output_dir / "result.json", "w") as f:
        json.dump(conv_result.model_dump(mode="json"), f, indent=2, default=str)

    with open(record_output_dir / "audit_log.json") as f:
        audit_log_data = json.load(f)

    write_trace(record_output_dir, record, audit_log_data, end_reason, turn_count, max_turns)
    logger.info(f"Trace saved to {record_output_dir / 'trace.txt'}")

    # ---- Build MetricContext ----
    turns_with_tools = build_conversation_trace(audit_log_data)
    assistant_turns, user_turns = build_per_role_turns(turns_with_tools)
    tool_params, tool_responses = extract_tool_params_and_responses(turns_with_tools)

    agent_instructions = raw_agent_config.get("instructions", "")
    if record.agent_override and record.agent_override.instructions:
        agent_instructions = record.agent_override.instructions

    # In text-only mode there is no TTS/STT — intended text == transcribed text
    conversation_finished = end_reason in ("goodbye", "transfer")

    metric_context = MetricContext(
        record_id=record.id,
        user_goal=record.user_goal,
        user_persona=record.user_config.get("user_persona", ""),
        expected_scenario_db=record.ground_truth.expected_scenario_db,
        initial_scenario_db=initial_db,
        final_scenario_db=final_db,
        initial_scenario_db_hash=initial_hash,
        final_scenario_db_hash=final_hash,
        agent_role=raw_agent_config.get("role", ""),
        agent_instructions=agent_instructions,
        agent_tools=raw_agent_config.get("tools", []),
        current_date_time=record.current_date_time,
        num_turns=stats["num_turns"],
        num_tool_calls=stats["num_tool_calls"],
        tools_called=stats["tools_called"],
        conversation_ended_reason=end_reason,
        duration_seconds=duration,
        output_dir=str(record_output_dir),
        conversation_trace=turns_with_tools,
        transcribed_assistant_turns=assistant_turns,
        transcribed_user_turns=user_turns,
        intended_assistant_turns=assistant_turns,  # text-only: LLM response = intended text
        intended_user_turns=user_turns,  # text-only: LLM response = intended text
        num_assistant_turns=len(assistant_turns),
        num_user_turns=len(user_turns),
        tool_params=tool_params,
        tool_responses=tool_responses,
        conversation_finished=conversation_finished,
    )

    # ---- Run metrics ----
    logger.info(f"Running metrics: {', '.join(requested_metrics)}")
    registry = get_global_registry()
    metric_scores: dict[str, MetricScore] = {}

    for metric_name in requested_metrics:
        try:
            metric = registry.create(metric_name)
            if metric is None:
                logger.warning(f"  {metric_name}: SKIP (not found in registry)")
                continue
            score = await metric.compute(metric_context)
            metric_scores[metric_name] = score
            status = "PASS" if (score.normalized_score or 0) >= 0.5 else "FAIL"
            logger.info(f"  {metric_name}: {score.score} (normalized: {score.normalized_score}) [{status}]")
            if score.details.get("diff_summary"):
                logger.info(f"    Diff: {score.details['diff_summary']}")
        except Exception as e:
            logger.error(f"Metric {metric_name} failed: {e}", exc_info=True)
            metric_scores[metric_name] = MetricScore(name=metric_name, score=0.0, normalized_score=0.0, error=str(e))
            logger.error(f"  {metric_name}: ERROR - {e}")

    # Compute response latency from conversation trace timestamps
    latency_score = compute_response_latency(audit_log_data.get("transcript", []))
    metric_scores["text_response_latency"] = latency_score
    logger.info(f"  text_response_latency: {latency_score.score}s (normalized: {latency_score.normalized_score})")

    record_metrics = RecordMetrics(
        record_id=record.id,
        context=metric_context.to_dict(),
        metrics=metric_scores,
    )
    (record_output_dir / "metrics.json").write_text(record_metrics.model_dump_json(indent=2))
    logger.info(f"Metrics saved to {record_output_dir / 'metrics.json'}")

    agentic_system.save_agent_perf_stats()

    # Compute EVA composite aggregates for this record
    record_metrics.aggregate_metrics = compute_record_aggregates(record_metrics, composites)
    (record_output_dir / "metrics.json").write_text(record_metrics.model_dump_json(indent=2))

    return {
        "record_id": record.id,
        "output_id": output_id,
        "end_reason": end_reason,
        "turns": turn_count,
        "duration": duration,
        "metrics": {k: v.model_dump(mode="json") for k, v in metric_scores.items()},
        "record_metrics": record_metrics,
    }


def _should_retry(result: dict[str, Any]) -> str | None:
    """Return a reason string if the result warrants a retry, else None."""
    if result.get("end_reason") == "max_turns":
        return "user simulator did not call end_call (max_turns reached)"
    ubf = result.get("metrics", {}).get("user_behavioral_fidelity", {})
    if ubf and ubf.get("normalized_score") == 0.0 and ubf.get("error") is None:
        return "user_behavioral_fidelity scored 1 (worst)"
    return None


def _archive_failed_attempt(output_dir: Path, output_id: str, attempt_number: int) -> None:
    """Move a failed attempt directory to an archive name."""
    record_dir = output_dir / output_id
    if not record_dir.exists():
        return
    archive_dir = output_dir / f"{output_id}_failed_attempt_{attempt_number}"
    while archive_dir.exists():
        attempt_number += 1
        archive_dir = output_dir / f"{output_id}_failed_attempt_{attempt_number}"
    shutil.move(str(record_dir), str(archive_dir))
    logger.debug(f"Archived {output_id} attempt {attempt_number} to {archive_dir}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only test flow for EVA")
    parser.add_argument("--record-id", help="Record ID to test (omit to run all records)")
    parser.add_argument(
        "--domain", default=os.getenv("EVA_DOMAIN", ""), help="Domain name (default: EVA_DOMAIN env var)"
    )
    parser.add_argument("--llm-model", help="LLM model name (default: LLM_MODEL env var)")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns (default: 20)")
    parser.add_argument("--max-attempts", type=int, default=1, help="Max attempts per record (default: 3)")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials per record for pass@k (default: 1)")
    parser.add_argument("--output-dir", default="debug_output", help="Output directory (default: debug_output)")
    parser.add_argument(
        "--accuracy-metrics",
        default=",".join(DEFAULT_ACCURACY_METRICS),
        help=f"Metrics for EVA-A composite (default: {','.join(DEFAULT_ACCURACY_METRICS)})",
    )
    parser.add_argument(
        "--experience-metrics",
        default=",".join(DEFAULT_EXPERIENCE_METRICS),
        help=f"Metrics for EVA-X composite (default: {','.join(DEFAULT_EXPERIENCE_METRICS)})",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(TEXT_COMPATIBLE_METRICS),
        help=f"Comma-separated metrics to run. Available: {', '.join(TEXT_COMPATIBLE_METRICS)}",
    )
    args = parser.parse_args()

    domain = args.domain
    if not domain:
        sys.exit("Error: --domain or EVA_DOMAIN env var is required.")

    dataset_path, scenario_db_dir, agent_config_path = resolve_paths(domain)

    # Parse requested metrics — always include user_behavioral_fidelity for validation
    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in requested_metrics:
        if m not in TEXT_COMPATIBLE_METRICS:
            sys.exit(f"Error: metric '{m}' is not text-compatible. Available: {', '.join(TEXT_COMPATIBLE_METRICS)}")
    if "user_behavioral_fidelity" not in requested_metrics:
        requested_metrics.append("user_behavioral_fidelity")

    # Parse EVA composite configuration
    accuracy_metrics = [m.strip() for m in args.accuracy_metrics.split(",") if m.strip()]
    experience_metrics = [m.strip() for m in args.experience_metrics.split(",") if m.strip()]

    # Build thresholds: reuse defaults for known metrics, default to (">=", 0.5) for others
    accuracy_thresholds = {m: DEFAULT_ACCURACY_THRESHOLDS.get(m, (">=", 0.5)) for m in accuracy_metrics}
    experience_thresholds = {m: DEFAULT_EXPERIENCE_THRESHOLDS.get(m, (">=", 0.5)) for m in experience_metrics}
    text_composites = build_text_composites(
        accuracy_metrics,
        accuracy_thresholds,
        experience_metrics,
        experience_thresholds,
    )
    logger.info(f"EVA-A components: {accuracy_metrics}")
    logger.info(f"EVA-X components: {experience_metrics}")

    # ---- Load data ----
    all_records = EvaluationRecord.load_dataset(dataset_path)

    if args.record_id:
        records = [r for r in all_records if r.id == args.record_id]
        if not records:
            sys.exit(f"Error: record '{args.record_id}' not found in dataset.")
    else:
        records = all_records

    agent = AgentConfig.from_yaml(agent_config_path)

    with open(agent_config_path) as f:
        raw_agent_config = yaml.safe_load(f)
    if isinstance(raw_agent_config, dict) and "agents" in raw_agent_config:
        raw_agent_config = raw_agent_config["agents"][0]

    tool_module = agent.tool_module_path
    if not tool_module:
        sys.exit("Error: tool_module_path not set in agent YAML config.")

    llm_model = args.llm_model or os.getenv("EVA_MODEL__LLM")
    if not llm_model:
        sys.exit("Error: --llm-model or EVA_MODEL__LLM env var is required.")

    # Initialize the LiteLLM Router from EVA_MODEL_LIST
    model_list_json = os.getenv("EVA_MODEL_LIST")
    if not model_list_json:
        sys.exit("Error: EVA_MODEL_LIST env var is required.")
    router.init(json.loads(model_list_json))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = llm_model.replace("/", "-")
    output_dir = Path(args.output_dir) / f"{timestamp}_{safe_model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_trials = args.num_trials
    max_attempts = args.max_attempts

    # Build output_id → record mapping (supports pass@k trials)
    output_id_to_record: dict[str, EvaluationRecord] = {}
    for record in records:
        if num_trials > 1:
            for trial_idx in range(num_trials):
                output_id_to_record[f"{record.id}/trial_{trial_idx}"] = record
        else:
            output_id_to_record[record.id] = record

    all_output_ids = list(output_id_to_record.keys())
    pending_output_ids = list(all_output_ids)
    rerun_history: dict[str, list[dict]] = {}

    logger.info(
        f"Running {len(records)} record(s) x {num_trials} trial(s) = {len(all_output_ids)} tasks "
        f"| domain={domain} | model={llm_model} | max_attempts={max_attempts}"
    )

    # ---- Attempt loop: run → validate → archive failures → retry ----
    async def _run_one(oid: str) -> dict[str, Any]:
        record = output_id_to_record[oid]
        try:
            return await run_record(
                record=record,
                agent=agent,
                raw_agent_config=raw_agent_config,
                tool_module=tool_module,
                agent_config_path=agent_config_path,
                scenario_db_dir=scenario_db_dir,
                llm_model=llm_model,
                output_dir=output_dir,
                max_turns=args.max_turns,
                requested_metrics=requested_metrics,
                output_id=oid,
                composites=text_composites,
            )
        except Exception as e:
            logger.error(f"{oid} failed: {e}", exc_info=True)
            return {"record_id": record.id, "output_id": oid, "error": str(e), "metrics": {}}

    final_results: dict[str, dict[str, Any]] = {}

    for attempt_number in range(1, max_attempts + 1):
        if not pending_output_ids:
            break

        logger.info(
            f"\n{'=' * 60}\n"
            f"Attempt {attempt_number}/{max_attempts}: running {len(pending_output_ids)} tasks\n"
            f"{'=' * 60}"
        )

        tasks = [_run_one(oid) for oid in pending_output_ids]
        results = await tqdm.gather(*tasks, desc=f"Attempt {attempt_number}", total=len(tasks))
        result_by_oid = {r.get("output_id", r.get("record_id")): r for r in results}

        # Validate results
        failed_this_attempt: list[str] = []
        for oid in pending_output_ids:
            result = result_by_oid.get(oid)
            if result is None or "error" in result and "end_reason" not in result:
                reason = result.get("error", "unknown error") if result else "no result"
                rerun_history.setdefault(oid, []).append({"attempt": attempt_number, "reason": f"error: {reason}"})
                failed_this_attempt.append(oid)
                final_results[oid] = result or {"record_id": "?", "output_id": oid, "error": "no result"}
                continue

            retry_reason = _should_retry(result)
            if retry_reason:
                rerun_history.setdefault(oid, []).append({"attempt": attempt_number, "reason": retry_reason})
                failed_this_attempt.append(oid)
            final_results[oid] = result

        # Archive and prepare for next attempt
        if failed_this_attempt and attempt_number < max_attempts:
            logger.info(f"{len(failed_this_attempt)} tasks failed validation — archiving for rerun...")
            for oid in failed_this_attempt:
                _archive_failed_attempt(output_dir, oid, attempt_number)
            pending_output_ids = failed_this_attempt
        elif failed_this_attempt:
            logger.warning(f"{len(failed_this_attempt)} tasks still failing after {max_attempts} attempts")
            pending_output_ids = []
        else:
            logger.info("All tasks passed validation!")
            pending_output_ids = []

    # ---- Summary ----
    all_results = [final_results[oid] for oid in all_output_ids if oid in final_results]

    logger.info(f"\nSUMMARY: {len(all_results)}/{len(all_output_ids)} tasks completed")
    for r in all_results:
        oid = r.get("output_id", r.get("record_id"))
        if "error" in r and "end_reason" not in r:
            logger.error(f"  {oid}: ERROR - {r['error']}")
        else:
            ms = ", ".join(f"{k}={v.get('normalized_score', '?')}" for k, v in r.get("metrics", {}).items())
            logger.info(f"  {oid}: {r.get('end_reason', '?')} ({r.get('turns', '?')} turns) | {ms}")

    # Save per-record results
    with open(output_dir / "summary.json", "w") as f:
        serializable = [{k: v for k, v in r.items() if k != "record_metrics"} for r in all_results]
        json.dump(serializable, f, indent=2, default=str)

    if rerun_history:
        with open(output_dir / "rerun_history.json", "w") as f:
            json.dump(rerun_history, f, indent=2)

    # ---- Metrics summary (across all records/trials) ----
    all_record_metrics: dict[str, RecordMetrics] = {}
    for r in all_results:
        rm = r.get("record_metrics")
        if isinstance(rm, RecordMetrics):
            all_record_metrics[r.get("output_id", r["record_id"])] = rm

    if all_record_metrics:
        metric_names = requested_metrics + ["text_response_latency"]
        per_metric = MetricsRunner._build_per_metric_aggregates(all_record_metrics, metric_names, num_draws=num_trials)

        # Augment text_response_latency with raw latency stats in seconds
        if "text_response_latency" in per_metric:
            raw_latencies = []
            all_turn_latencies: list[float] = []
            turn_latencies_with_tools: list[float] = []
            turn_latencies_without_tools: list[float] = []
            all_call_latencies: list[float] = []
            for rm in all_record_metrics.values():
                ms = rm.metrics.get("text_response_latency")
                if ms is None or ms.error is not None or ms.score is None:
                    continue
                raw_latencies.append(ms.score)
                for turn in (ms.details or {}).get("per_turn", []):
                    lat = turn["latency_s"]
                    all_turn_latencies.append(lat)
                    has_tools = any(c.get("type") == "tool" for c in turn.get("calls", []))
                    if has_tools:
                        turn_latencies_with_tools.append(lat)
                    else:
                        turn_latencies_without_tools.append(lat)
                    for call in turn.get("calls", []):
                        if call.get("type") == "llm":
                            all_call_latencies.append(call["latency_s"])
            if raw_latencies:
                per_metric["text_response_latency"]["mean_latency_s"] = round(
                    sum(raw_latencies) / len(raw_latencies), 3
                )
                per_metric["text_response_latency"]["min_latency_s"] = round(min(raw_latencies), 3)
                per_metric["text_response_latency"]["max_latency_s"] = round(max(raw_latencies), 3)
            if all_turn_latencies:
                per_metric["text_response_latency"]["mean_turn_latency_s"] = round(
                    sum(all_turn_latencies) / len(all_turn_latencies), 3
                )
            if turn_latencies_with_tools:
                per_metric["text_response_latency"]["mean_turn_latency_s_with_tools"] = round(
                    sum(turn_latencies_with_tools) / len(turn_latencies_with_tools), 3
                )
            if turn_latencies_without_tools:
                per_metric["text_response_latency"]["mean_turn_latency_s_without_tools"] = round(
                    sum(turn_latencies_without_tools) / len(turn_latencies_without_tools), 3
                )
            if all_call_latencies:
                per_metric["text_response_latency"]["mean_llm_call_latency_s"] = round(
                    sum(all_call_latencies) / len(all_call_latencies), 3
                )

        overall_scores = compute_run_level_aggregates(all_record_metrics, num_trials, text_composites)
        data_quality = MetricsRunner._build_data_quality(all_record_metrics, per_metric)

        metrics_summary_data: dict[str, Any] = {
            "total_records": len(all_record_metrics),
            "num_trials": num_trials,
            "max_attempts": max_attempts,
            "data_quality": data_quality,
            "overall_scores": overall_scores,
            "per_metric": per_metric,
        }

        if rerun_history:
            metrics_summary_data["rerun_summary"] = {
                "records_retried": len(rerun_history),
                "total_retries": sum(len(v) for v in rerun_history.values()),
            }

        summary_path = output_dir / "metrics_summary.json"
        summary_path.write_text(json.dumps(metrics_summary_data, indent=2))
        logger.info(f"Metrics summary saved to {summary_path}")
        logger.info(json.dumps(metrics_summary_data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
