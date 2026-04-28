"""Voice agent metrics postprocessor - processes logs to create metric variables."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from eva.assistant.agentic.system import GENERIC_ERROR
from eva.models.config import PipelineType
from eva.models.results import ConversationResult
from eva.utils.log_processing import (
    AnnotationLabel,
    aggregate_pipecat_logs_by_type,
    align_turn_keys,
    annotate_last_entry,
    append_turn_text,
    extract_tool_params_and_responses,
    filter_empty_responses,
    get_entry_for_audit_log,
    group_consecutive_turns,
    truncate_to_spoken,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)


def last_audio_speaker(
    audio_timestamps_user_turns: dict[int, list[tuple[float, float]]],
    audio_timestamps_assistant_turns: dict[int, list[tuple[float, float]]],
) -> str | None:
    """Return the role whose audio ended latest, or None if neither recorded audio."""

    def _latest_end(intervals_by_turn: dict[int, list[tuple[float, float]]]) -> float | None:
        ends = [iv[1] for intervals in intervals_by_turn.values() if intervals for iv in intervals]
        return max(ends) if ends else None

    user_end = _latest_end(audio_timestamps_user_turns)
    asst_end = _latest_end(audio_timestamps_assistant_turns)
    if user_end is None and asst_end is None:
        return None
    if user_end is None:
        return "assistant"
    if asst_end is None:
        return "user"
    return "user" if user_end > asst_end else "assistant"


def is_agent_timeout_on_user_turn(
    conversation_ended_reason: str | None,
    audio_timestamps_user_turns: dict[int, list[tuple[float, float]]],
    audio_timestamps_assistant_turns: dict[int, list[tuple[float, float]]],
) -> bool:
    """True if conversation ended with inactivity_timeout and the user spoke last."""
    if conversation_ended_reason != "inactivity_timeout":
        return False
    return last_audio_speaker(audio_timestamps_user_turns, audio_timestamps_assistant_turns) == "user"


def _resolve_path(stored: str | None, output_dir: Path) -> str | None:
    """Return *stored* if it exists on disk, otherwise ``output_dir / basename(stored)``.

    Allows metrics to re-run correctly when a run directory has been moved:
    *stored* reflects the original location, but the file is now under *output_dir*
    with the same filename. Returns ``None`` when *stored* is ``None`` so callers
    can treat ``None`` as "feature disabled" (e.g. audio recording was off).
    """
    if stored is None:
        return None
    if Path(stored).exists():
        return stored
    return str(output_dir / Path(stored).name)


# Elevenlabs audio user field → _ProcessorContext attribute name
AUDIO_ATTR = {
    "framework_agent": "audio_timestamps_assistant_turns",
    "pipecat_agent": "audio_timestamps_assistant_turns",  # Deprecated name - kept for old runs
    "elevenlabs_user": "audio_timestamps_user_turns",
}

# Turn variable names grouped by role, used for cross-source alignment checks
TURN_VARS_BY_ROLE = {
    "assistant": [
        "intended_assistant_turns",
        "transcribed_assistant_turns",
    ],
    "user": [
        "transcribed_user_turns",
        "intended_user_turns",
    ],
}


@dataclass
class _TurnExtractionState:
    """Mutable state for the single-pass event loop in _extract_turns_from_history.

    Turns are numbered by ElevenLabs ``audio_start(elevenlabs_user)`` events.
    Turn 0 = assistant greeting (before first user audio).
    """

    turn_num: int = 0  # Turn counter (0 = greeting, incremented by user events)
    assistant_spoke_in_turn: bool = False  # Assistant has spoken since last user event
    user_audio_started_in_turn: bool = False  # User audio started in the current turn
    assistant_processed_in_turn: bool = False  # Tool calls happened in the current turn
    hold_turn: bool = False  # After an interruption, hold the turn for one advance cycle
    audio_starts: dict[tuple[str, int], list[float]] = field(default_factory=dict)
    audio_ends: dict[tuple[str, int], list[float]] = field(default_factory=dict)
    last_audio_start_key: dict[str, tuple[str, int]] = field(default_factory=dict)
    session_end_ts: float | None = None
    user_audio_open: bool = False
    assistant_audio_open: bool = False
    assistant_interrupted_turns: set[int] = field(default_factory=set)
    user_interrupted_turns: set[int] = field(default_factory=set)
    pending_user_interrupts_label: bool = False  # Next user entry should get [user interrupts] prefix
    # Track which turn each speaker's audio started at, so late-arriving speech transcripts land at the correct turn.
    last_assistant_audio_turn: int = 0
    last_user_audio_turn: int = 0
    # True when framework_agent audio started after user audio ended, meaning any subsequent user_speech (while
    # user_audio_open is False) belongs to a new speaking session and should be buffered until the next
    # audio_start(elevenlabs_user) sets the correct turn.
    assistant_responded_since_user_ended: bool = False
    # True when user_speech was received in the current ElevenLabs user audio session. If False at the next
    # audio_start(elevenlabs_user), the previous session was empty and should not create a new turn.
    user_speech_in_session: bool = False
    # Buffer for user_speech events that arrive before the first audio_start(elevenlabs_user). Replayed once the
    # audio_start fires.
    buffered_user_speech: list[dict] = field(default_factory=list)
    # Track text of buffered user_speech to deduplicate post-audio_start copies.
    buffered_user_speech_texts: set[str] = field(default_factory=set)
    # Set on empty-session rollback so the next audio_start(elevenlabs_user) can advance even though
    # assistant_spoke_in_turn was consumed by the (now undone) advance.  Also checked by audit_log/user — if pipecat
    # captured speech that ElevenLabs missed, the user IS speaking and the audit_log/user should advance to a new turn.
    pending_advance_after_rollback: bool = False
    # True when an audit_log/user consumed pending_advance_after_rollback. At the next audio_start(elevenlabs_user), if
    # the user is interrupting the assistant (assistant_audio_open), this is the same utterance — skip the advance so
    # user_speech lands at the same turn.
    rollback_advance_consumed_by_user: bool = False

    def advance_turn_if_needed(self, from_audio_start: bool = False, bypass_hold: bool = False) -> None:
        """Advance turn if the assistant responded since the last user event.

        Called on audio_start(elevenlabs_user) and audit_log/user events.
        After an interruption, hold_turn suppresses one advance from audit_log/user
        (late STT from the interrupted session) but never blocks audio_start
        (the user speaking again always starts a new turn).

        bypass_hold=True is used by S2S pipelines, where audit_log/user carries the
        S2S model's own transcription of the current utterance, not a late STT chunk
        from the previous (interrupted) session.
        """
        if self.hold_turn and not bypass_hold:
            if from_audio_start:
                # New user speech — clear hold_turn but still advance
                self.hold_turn = False
            else:
                # Late STT chunk from interrupted session — consume without advancing
                self.hold_turn = False
                self.assistant_spoke_in_turn = False
                return
        if self.assistant_spoke_in_turn:
            self.turn_num += 1
            self.assistant_spoke_in_turn = False
            self.user_audio_started_in_turn = False
            self.assistant_processed_in_turn = False


def _user_transcript_separator(existing: str, turn: int, state: _TurnExtractionState) -> str:
    """Return the separator for transcribed_user_turns between consecutive user chunks."""
    if not existing:
        return ""
    if turn in state.assistant_interrupted_turns:
        return f" {AnnotationLabel.ASSISTANT_INTERRUPTS} "
    return " "


def _assistant_speech_separator(existing: str, turn: int, state: _TurnExtractionState) -> str:
    """Return the separator for transcribed_assistant_turns between consecutive speech chunks."""
    if not existing:
        return ""
    if turn in state.user_interrupted_turns:
        return f" {AnnotationLabel.USER_INTERRUPTS} "
    if turn in state.assistant_interrupted_turns:
        return f" {AnnotationLabel.CUT_OFF_ON_ITS_OWN} "
    return f" {AnnotationLabel.LIKELY_INTERRUPTION} "


def _process_user_speech(
    event: dict,
    state: _TurnExtractionState,
    context: "_ProcessorContext",
    conversation_trace: list[dict],
    pipeline_type: PipelineType,
) -> None:
    """Process a single user_speech event into intended_user_turns (and audio-native trace)."""
    turn_idx = state.last_user_audio_turn
    existing = context.intended_user_turns.get(turn_idx, "")
    sep = f" {AnnotationLabel.CUT_OFF_ON_ITS_OWN} " if existing else ""
    user_text = event["data"]["data"]["text"]
    if not existing and state.pending_user_interrupts_label:
        user_text = f"{AnnotationLabel.USER_INTERRUPTS} {user_text}"
    append_turn_text(context.intended_user_turns, turn_idx, user_text, sep)
    state.user_speech_in_session = True
    # For audio-native models, use intended user text in the conversation trace
    if pipeline_type in (PipelineType.S2S, PipelineType.AUDIO_LLM):
        trace_entry = {
            "role": "user",
            "content": user_text,
            "timestamp": event["timestamp_ms"],
            "type": "intended",
            "turn_id": turn_idx,
        }
        if existing and turn_idx in state.user_interrupted_turns:
            trace_entry["content"] = f"{AnnotationLabel.USER_INTERRUPTS} {user_text}"
        elif state.pending_user_interrupts_label:
            trace_entry["content"] = f"{AnnotationLabel.USER_INTERRUPTS} {user_text}"
            state.pending_user_interrupts_label = False
        conversation_trace.append(trace_entry)


def _warn_turn_misalignment(context: "_ProcessorContext") -> None:
    """Log warnings if turn indices have gaps within any source's own range."""
    for role, var_names in TURN_VARS_BY_ROLE.items():
        populated = {name: set(getattr(context, name).keys()) for name in var_names if getattr(context, name)}
        if not populated:
            logger.warning(f"Record {context.record_id}: No populated turn variables for role '{role}'")
            continue

        for name, keys in populated.items():
            expected = set(range(min(keys), max(keys) + 1))
            gaps = sorted(expected - keys)
            if gaps:
                logger.warning(f"Record {context.record_id}: {name} has gaps at turns {gaps}")


def _handle_audit_log_event(
    event: dict,
    state: "_TurnExtractionState",
    context: "_ProcessorContext",
    conversation_trace: list[dict],
    pipeline_type: PipelineType,
) -> None:
    """Process a single audit_log source event into turn variables and conversation trace."""
    if event["event_type"] == "user":
        if state.pending_advance_after_rollback:
            # Pipecat captured speech that ElevenLabs missed during an empty session. The user IS speaking — advance.
            state.assistant_spoke_in_turn = True
            state.pending_advance_after_rollback = False
            state.rollback_advance_consumed_by_user = True
        # While user audio is active, suppress the turn increment (user is still speaking) but still call
        # advance so that hold_turn is consumed if set.
        if state.user_audio_open:
            state.assistant_spoke_in_turn = False
        state.advance_turn_if_needed(bypass_hold=pipeline_type == PipelineType.S2S)
        turn = state.turn_num
        entry = get_entry_for_audit_log(event, turn)
        existing = context.transcribed_user_turns.get(turn, "")
        # Prefix entry if this is a second user transcript after user interrupted
        if existing and turn in state.user_interrupted_turns:
            entry["content"] = f"{AnnotationLabel.USER_INTERRUPTS} {entry['content']}"
            state.pending_user_interrupts_label = False
        # Prefix if this is the first user entry after a user-interrupts-assistant advance
        elif state.pending_user_interrupts_label:
            entry["content"] = f"{AnnotationLabel.USER_INTERRUPTS} {entry['content']}"
            state.pending_user_interrupts_label = False
        # For audio-native models, user trace entries come from ElevenLabs user_speech instead
        if pipeline_type == PipelineType.CASCADE:
            conversation_trace.append(entry)
        sep = _user_transcript_separator(existing, turn, state)
        append_turn_text(context.transcribed_user_turns, turn, entry["content"], sep)

    elif event["event_type"] == "assistant":
        if pipeline_type == PipelineType.S2S:
            return
        turn = state.turn_num
        content = event["data"]
        # Apply interruption prefix if this is the first assistant entry in a turn where assistant barged in
        # on the user.
        if turn in state.assistant_interrupted_turns:
            has_prior = any(e.get("role") == "assistant" and e.get("turn_id") == turn for e in conversation_trace)
            if not has_prior:
                content = f"{AnnotationLabel.ASSISTANT_INTERRUPTS} {content}"
                user_entry_type = "transcribed" if pipeline_type == PipelineType.CASCADE else "intended"
                annotate_last_entry(
                    conversation_trace, turn, "user", user_entry_type, AnnotationLabel.CUT_OFF_BY_ASSISTANT
                )
        conversation_trace.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": event["timestamp_ms"],
                "type": "intended",
                "turn_id": turn,
                "_audit_source": True,
            }
        )

    elif event["event_type"] in ("tool_call", "tool_response"):
        state.assistant_processed_in_turn = True
        conversation_trace.append(get_entry_for_audit_log(event, state.turn_num))


def _handle_pipecat_event(
    event: dict,
    state: "_TurnExtractionState",
    context: "_ProcessorContext",
    conversation_trace: list[dict],
) -> None:
    """Process a single pipecat source event into intended_assistant_turns.

    Pipecat feeds intended_assistant_turns for metrics that need the full TTS text. The conversation trace
    uses audit_log/assistant entries (which preserve tool call boundaries), truncated in post-processing to
    only the portion that was actually spoken.
    """
    if event["event_type"] not in ("tts_text", "llm_response"):
        return
    state.assistant_spoke_in_turn = True
    turn = state.turn_num
    existing = context.intended_assistant_turns.get(turn, "")

    if existing:
        if turn in state.user_interrupted_turns:
            annotate_last_entry(conversation_trace, turn, "assistant", "intended", AnnotationLabel.CUT_OFF_BY_USER)
        elif turn in state.assistant_interrupted_turns:
            annotate_last_entry(conversation_trace, turn, "assistant", "intended", AnnotationLabel.CUT_OFF_ON_ITS_OWN)

    if not existing:
        sep = ""
    elif turn in state.user_interrupted_turns:
        sep = f" {AnnotationLabel.CUT_OFF_BY_USER} "
    elif state.assistant_processed_in_turn:
        sep = f" {AnnotationLabel.PAUSE_TOOL_CALL} "
    else:
        sep = f" {AnnotationLabel.CUT_OFF_ON_ITS_OWN} "
    text = event["data"]["frame"]
    if not existing and turn in state.assistant_interrupted_turns:
        text = f"{AnnotationLabel.ASSISTANT_INTERRUPTS} {text}"
    append_turn_text(context.intended_assistant_turns, turn, text, sep)
    # Also store raw segment for prefix matching during truncation
    context._intended_assistant_segments.setdefault(turn, []).append(text)

    # Pipeline-generated messages (e.g. generic error) have no audit log entry, so they would never
    # appear in the trace. Append them directly to keep ordering with tool calls.
    if text == GENERIC_ERROR:
        conversation_trace.append(
            {
                "role": "assistant",
                "content": text,
                "type": "intended",
                "turn_id": turn,
            }
        )


def _handle_audio_start(
    event: dict,
    state: "_TurnExtractionState",
    context: "_ProcessorContext",
    conversation_trace: list[dict],
    pipeline_type: PipelineType,
) -> None:
    """Process an ElevenLabs audio_start event, advancing the turn counter if needed."""
    role = event["data"]["user"]
    timestamp = event["data"]["audio_timestamp"]

    if role == "elevenlabs_user":
        if state.assistant_audio_open:
            # User interrupts assistant — apply "[likely cut off by user]" labels to the OLD turn now,
            # then advance so the user's retry starts a new turn.
            cut_turn = state.turn_num
            if cut_turn in context.intended_assistant_turns:
                context.intended_assistant_turns[cut_turn] += f" {AnnotationLabel.CUT_OFF_BY_USER}"
            if cut_turn in context.transcribed_assistant_turns:
                context.transcribed_assistant_turns[cut_turn] += f" {AnnotationLabel.CUT_OFF_BY_USER}"
            annotate_last_entry(conversation_trace, cut_turn, "assistant", "intended", AnnotationLabel.CUT_OFF_BY_USER)
            state.pending_user_interrupts_label = True
        state.user_speech_in_session = False
        if state.rollback_advance_consumed_by_user and state.assistant_audio_open:
            # An audit_log/user already advanced for this utterance (Deepgram caught speech during empty
            # sessions). The user is now interrupting the assistant's response — this is the same
            # speech, so don't advance again.
            state.rollback_advance_consumed_by_user = False
            state.assistant_spoke_in_turn = False
        elif state.pending_advance_after_rollback:
            # A previous empty session rolled back and deferred its advance. Force it now so this real
            # session starts at the correct (next) turn.
            state.assistant_spoke_in_turn = True
            state.pending_advance_after_rollback = False
        state.rollback_advance_consumed_by_user = False
        state.advance_turn_if_needed(from_audio_start=True)
        # Mark the NEW turn (after advance) as a user-interrupted turn — the user's interrupting speech
        # lands here, symmetric with assistant_interrupted_turns.
        if state.pending_user_interrupts_label:
            state.user_interrupted_turns.add(state.turn_num)
        state.user_audio_open = True
        state.user_audio_started_in_turn = True
        state.last_user_audio_turn = state.turn_num
        state.assistant_responded_since_user_ended = False
        # Replay any buffered user_speech that arrived before this audio_start — now we know the correct turn.
        if state.buffered_user_speech:
            for buffered in state.buffered_user_speech:
                _process_user_speech(buffered, state, context, conversation_trace, pipeline_type)
            state.buffered_user_speech.clear()

    elif role in ("framework_agent", "pipecat_agent"):
        state.assistant_audio_open = True
        state.last_assistant_audio_turn = state.turn_num
        if not state.user_audio_open:
            state.assistant_responded_since_user_ended = True
            # For S2S pipelines, mark that the assistant spoke as soon as audio starts.
            # EL assistant_speech transcripts arrive late (often at the same timestamp as
            # the next user's audio_start) and can't be relied on for turn boundary detection.
            if pipeline_type == PipelineType.S2S:
                state.assistant_spoke_in_turn = True
        # Interruption: assistant starts speaking while user is still speaking. Only count if (a) user
        # audio started in the current turn (not lingering from a previous turn's delayed delivery) and
        # (b) no tool calls happened yet — tool calls mean the assistant is responding to the user's
        # input, not barging in.
        if state.user_audio_open and state.user_audio_started_in_turn and not state.assistant_processed_in_turn:
            state.assistant_interrupted_turns.add(state.turn_num)
            state.hold_turn = True

    turn_idx = state.turn_num
    key = (role, turn_idx)
    state.last_audio_start_key[role] = key
    state.audio_starts.setdefault(key, []).append(timestamp)


def _handle_audio_end(event: dict, state: "_TurnExtractionState") -> None:
    """Process an ElevenLabs audio_end event, recording the end timestamp and closing the audio session."""
    role = event["data"]["user"]
    timestamp = event["data"]["audio_timestamp"]
    if (key := state.last_audio_start_key.get(role)) is not None:
        state.audio_ends.setdefault(key, []).append(timestamp)
    if role == "elevenlabs_user":
        state.user_audio_open = False
        # If the user audio session produced no user_speech, it was an empty burst (e.g. background
        # noise) that should not count as a turn. Roll back the turn advance that happened at this
        # session's audio_start so the next real session can advance normally.
        if not state.user_speech_in_session and state.user_audio_started_in_turn:
            state.turn_num -= 1
            state.user_audio_started_in_turn = False
            # Defer the advance to the next real audio_start(elevenlabs_user). Do NOT restore
            # assistant_spoke_in_turn — this prevents late audit_log/user STT chunks from advancing
            # (they naturally stay at the current turn).
            state.pending_advance_after_rollback = True
    elif role in ("framework_agent", "pipecat_agent"):
        state.assistant_audio_open = False


def _handle_elevenlabs_event(
    event: dict,
    state: "_TurnExtractionState",
    context: "_ProcessorContext",
    conversation_trace: list[dict],
    pipeline_type: PipelineType,
) -> bool:
    """Process a single elevenlabs source event. Returns True if the caller should continue."""
    if event["event_type"] == "assistant_speech":
        # Use the turn where assistant audio started, not the current turn — ElevenLabs transcripts can
        # arrive after a user audio_start has already advanced the turn.
        turn = state.last_assistant_audio_turn
        # Only mark "assistant spoke" if the speech belongs to the current turn; late transcripts from a
        # previous turn must not trigger a spurious turn advance.  For S2S pipelines, audio_start(framework_agent)
        # already sets assistant_spoke_in_turn, so the S2S-specific override here is no longer needed.
        if turn == state.turn_num:
            state.assistant_spoke_in_turn = True
        existing = context.transcribed_assistant_turns.get(turn, "")
        sep = _assistant_speech_separator(existing, turn, state)
        text = event["data"]["data"]["text"]
        if not existing and turn in state.assistant_interrupted_turns:
            text = f"{AnnotationLabel.ASSISTANT_INTERRUPTS} {text}"
        append_turn_text(context.transcribed_assistant_turns, turn, text, sep)
        # For S2S, assistant trace entries come from EL (audit log assistant entries are skipped)
        if pipeline_type == PipelineType.S2S:
            conversation_trace.append(
                {
                    "role": "assistant",
                    "content": text,
                    "timestamp": event["timestamp_ms"],
                    "type": "transcribed",
                    "turn_id": turn,
                }
            )

    elif event["event_type"] == "user_speech":
        # Buffer user_speech when it cannot be paired with the current user audio session. This happens when:
        # - The transcript arrives before the first audio_start
        #   (ElevenLabs sends speech slightly before audio_start)
        # - The assistant responded after the user's last audio ended, so this speech is for a NEW session
        #   whose audio_start hasn't arrived yet.
        # Late transcripts for the SAME session (arriving shortly after audio_end, before any new assistant
        # response) are NOT buffered — they use last_user_audio_turn directly.
        if not state.user_audio_open and state.assistant_responded_since_user_ended:
            state.buffered_user_speech.append(event)
            state.buffered_user_speech_texts.add(event["data"]["data"]["text"])
            state.user_speech_in_session = True
            return True  # signal "continue" to caller
        # Deduplicate: skip if this is a post-audio_start copy of a buffered event (ElevenLabs sometimes
        # sends it twice).
        raw_text = event["data"]["data"]["text"]
        if raw_text in state.buffered_user_speech_texts:
            state.buffered_user_speech_texts.discard(raw_text)
            return False
        _process_user_speech(event, state, context, conversation_trace, pipeline_type)

    elif event["event_type"] == "audio_start":
        _handle_audio_start(event, state, context, conversation_trace, pipeline_type)

    elif event["event_type"] == "audio_end":
        _handle_audio_end(event, state)

    elif event["event_type"] == "connection_state":
        if event["data"]["data"]["state"] == "session_ended":
            state.session_end_ts = event["timestamp_ms"] / 1000.0

    return False


def _pair_audio_segments(state: "_TurnExtractionState", context: "_ProcessorContext") -> None:
    """Pair audio_start/audio_end lists into (start, end) tuples per turn.

    If an audio_start has no matching audio_end, session_end_ts is used as fallback.
    """
    for (role, turn_idx), starts in state.audio_starts.items():
        ends = state.audio_ends.get((role, turn_idx), [])
        segments: list[tuple[float, float]] = []
        for i, s in enumerate(starts):
            if i < len(ends):
                segments.append((s, ends[i]))
            elif state.session_end_ts is not None:
                segments.append((s, state.session_end_ts))
        if segments:
            getattr(context, AUDIO_ATTR[role])[turn_idx] = segments


def _validate_conversation_trace(
    conversation_trace: list[dict],
    context: "_ProcessorContext",
) -> list[dict]:
    """Validate audit-log assistant entries against pipecat text.

    The audit log records the full LLM response, but only the portion sent to TTS was actually spoken.
    Truncates entries to the spoken prefix; drops entries with no overlap (never spoken at all).
    """
    validated_trace = []
    for entry in conversation_trace:
        if not entry.get("_audit_source"):
            validated_trace.append(entry)
            continue
        turn_id = entry.get("turn_id")
        pipecat_segments = context._intended_assistant_segments.get(turn_id, [])
        audit_text = entry.get("content", "")
        truncated = truncate_to_spoken(audit_text, pipecat_segments)
        if truncated is not None:
            if truncated != audit_text:
                logger.warning(
                    f"Record {context.record_id}: Truncated assistant text "
                    f"at turn {turn_id}/{len(context.intended_assistant_turns)}: {audit_text[:80]!r} -> {truncated[:80]!r}"
                )
            entry["content"] = truncated
            entry.pop("_audit_source")
            validated_trace.append(entry)
        else:
            logger.warning(
                f"Record {context.record_id}: Filtered unsaid assistant text "
                f"at turn {turn_id}/{len(context.intended_assistant_turns)}: {audit_text[:80]!r}"
            )
    return validated_trace


def _fix_interruption_labels(context: "_ProcessorContext", state: "_TurnExtractionState") -> None:
    """Fix interruption labels that may have been missed during the event loop.

    The audit_log/assistant entry can arrive before the interruption is detected at audio_start(framework_agent),
    so the prefix wasn't applied during the loop. Only fix the first assistant entry per interrupted turn.
    """
    # Clean up per-entry interrupted keys (used during event loop only)
    for entry in context.conversation_trace:
        entry.pop("interrupted", None)

    # Fix [assistant interrupts] labels
    labeled_asst_turns: set[int] = set()
    for entry in context.conversation_trace:
        if entry.get("role") != "assistant":
            continue
        tid = entry.get("turn_id")
        if tid not in state.assistant_interrupted_turns or tid in labeled_asst_turns:
            continue
        labeled_asst_turns.add(tid)
        if not entry["content"].startswith(AnnotationLabel.ASSISTANT_INTERRUPTS):
            entry["content"] = f"{AnnotationLabel.ASSISTANT_INTERRUPTS} {entry['content']}"

    # Fix [user interrupts] labels. Unlike [assistant interrupts], the event loop usually handles this
    # (audio_start arrives before audit_log/user). Only add the label when no user entry at the interrupted
    # turn already carries it — avoids mislabeling the first entry in no-advance (rollback) cases where
    # the original speech precedes the interrupting speech at the same turn.
    for tid in state.user_interrupted_turns:
        user_entries = [e for e in context.conversation_trace if e.get("role") == "user" and e.get("turn_id") == tid]
        already_labeled = any(e["content"].startswith(AnnotationLabel.USER_INTERRUPTS) for e in user_entries)
        if not already_labeled and user_entries:
            user_entries[0]["content"] = f"{AnnotationLabel.USER_INTERRUPTS} {user_entries[0]['content']}"


def _finalize_extraction(
    context: "_ProcessorContext",
    state: "_TurnExtractionState",
    conversation_trace: list[dict],
) -> None:
    """Assign derived context variables, log results, and align turn keys across all sources."""
    context.assistant_interrupted_turns = state.assistant_interrupted_turns
    context.user_interrupted_turns = state.user_interrupted_turns

    all_interrupted = state.assistant_interrupted_turns | state.user_interrupted_turns
    if all_interrupted:
        logger.info(
            f"Record {context.record_id}: Detected interruptions — "
            f"assistant interrupted user at turns {sorted(state.assistant_interrupted_turns)}, "
            f"user interrupted assistant at turns {sorted(state.user_interrupted_turns)}"
        )
    context.tool_params, context.tool_responses = extract_tool_params_and_responses(conversation_trace)
    context.tool_called = [t["tool_name"].lower() for t in context.tool_params]
    context.num_tool_calls = len(context.tool_params)
    if context.pipeline_type == PipelineType.S2S:
        context.num_assistant_turns = len(context.transcribed_assistant_turns)
    else:
        context.num_assistant_turns = len(context.intended_assistant_turns)
    context.num_user_turns = len(context.transcribed_user_turns)

    _warn_turn_misalignment(context)

    # Ensure all per-role dicts share the same keys, filling missing entries with defaults so downstream
    # metrics don't need to handle missing keys.
    align_turn_keys(
        context.transcribed_user_turns,
        context.intended_user_turns,
        context.audio_timestamps_user_turns,
    )
    align_turn_keys(
        context.transcribed_assistant_turns,
        context.intended_assistant_turns,
        context.audio_timestamps_assistant_turns,
    )

    logger.info(
        f"Record {context.record_id}: Extracted turns - "
        f"{context.num_assistant_turns} assistant (keys {sorted(context.intended_assistant_turns.keys())}), "
        f"{context.num_user_turns} user (keys {sorted(context.transcribed_user_turns.keys())})"
    )


def _ensure_greeting_is_first(context: "_ProcessorContext") -> None:
    """Ensure the assistant greeting (turn 0) is the first entry in conversation_trace.

    With audio-native models, a ElevenLabs user_speech timestamp can arrive before the audit-log assistant entry, so the
    greeting ends up out of order. Move it to the front, or synthesize it from pipecat text if absent.
    """
    first = context.conversation_trace[0]
    if not (first.get("role") == "user" and first.get("turn_id", 0) > 0):
        return

    greeting_idx = next(
        (i for i, e in enumerate(context.conversation_trace) if e.get("role") == "assistant" and e.get("turn_id") == 0),
        None,
    )
    if greeting_idx is not None:
        greeting = context.conversation_trace.pop(greeting_idx)
    else:
        # Greeting not in audit log — create from pipecat text (cascade) or transcribed text (S2S).
        greeting_text = context.intended_assistant_turns.get(0) or context.transcribed_assistant_turns.get(0)
        greeting = {
            "role": "assistant",
            "content": greeting_text,
            "type": "intended" if context.intended_assistant_turns.get(0) else "transcribed",
            "turn_id": 0,
        }
    context.conversation_trace.insert(0, greeting)


def _label_trailing_assistant_turn(context: "_ProcessorContext", last_entry: dict, last_turn_id: int) -> None:
    """Label the last assistant turn with CUT_OFF_ON_ITS_OWN and sync across trace/intended/transcribed.

    Two sub-cases:
      a) Trace already ends with an assistant entry — update content in place.
      b) Trace ends with a user entry but intended_assistant_turns has content at that turn or later
         with no trace entry — append a new entry.
    """
    trailing_turn_id: int | None = None
    if last_entry.get("role") == "assistant":
        trailing_turn_id = last_turn_id
    elif context.intended_assistant_turns:
        max_asst = max(context.intended_assistant_turns.keys())
        if max_asst >= last_turn_id:
            has_asst_in_trace = any(
                e.get("role") == "assistant" and e.get("turn_id") == max_asst for e in context.conversation_trace
            )
            if not has_asst_in_trace and context.intended_assistant_turns[max_asst]:
                trailing_turn_id = max_asst

    if trailing_turn_id is None:
        return

    text = (
        context.conversation_trace[-1]["content"]
        if last_entry.get("role") == "assistant"
        else context.intended_assistant_turns[trailing_turn_id]
    )
    labeled = f"{text} {AnnotationLabel.CUT_OFF_ON_ITS_OWN}"

    if last_entry.get("role") == "assistant":
        context.conversation_trace[-1]["content"] = labeled
    else:
        context.conversation_trace.append(
            {"role": "assistant", "content": labeled, "type": "intended", "turn_id": trailing_turn_id}
        )

    # Sync intended + transcribed (skip intended for S2S — no intended text exists)
    if context.pipeline_type != PipelineType.S2S:
        context.intended_assistant_turns[trailing_turn_id] = labeled
    if not context.transcribed_assistant_turns.get(trailing_turn_id):
        context.transcribed_assistant_turns[trailing_turn_id] = labeled
    else:
        context.transcribed_assistant_turns[trailing_turn_id] += f" {AnnotationLabel.CUT_OFF_ON_ITS_OWN}"

    logger.info(f"Record {context.record_id}: Labeled trailing assistant at turn {trailing_turn_id}")


class _ProcessorContext:
    """Processed log data for metric computation."""

    def __init__(self):
        self.record_id: str | None = None

        # Per-role turn data (indexed by turn_id, 0-indexed)
        self.transcribed_assistant_turns: dict[int, str] = {}
        self.transcribed_user_turns: dict[int, str] = {}
        self.intended_assistant_turns: dict[int, str] = {}
        self.intended_user_turns: dict[int, str] = {}

        # Raw TTS segments per turn, used for prefix matching during truncation validation.
        self._intended_assistant_segments: dict[int, list[str]] = {}
        self.audio_timestamps_assistant_turns: dict[int, list[tuple[float, float]]] = {}
        self.audio_timestamps_user_turns: dict[int, list[tuple[float, float]]] = {}

        self.num_assistant_turns: int = 0
        self.num_user_turns: int = 0
        self.num_tool_calls: int = 0

        self.tool_called: list[str] = []
        self.tool_params: list[dict] = []
        self.tool_responses: list[dict] = []

        self.conversation_trace: list[dict] = []

        self.audio_assistant_path: str | None = None
        self.audio_user_path: str | None = None
        self.audio_mixed_path: str | None = None

        # Interruption data
        self.assistant_interrupted_turns: set[int] = set()
        self.user_interrupted_turns: set[int] = set()

        # Conversation metadata
        self.conversation_ended_reason: str | None = None
        self.pipeline_type: PipelineType = PipelineType.CASCADE

        # Per-turn latency: user_end -> assistant_start (seconds)
        self.latency_assistant_turns: dict[int, float] = {}

        # Unified timeline of all events from all log sources
        self.history: list[dict] = []


class MetricsContextProcessor:
    """Postprocessor for voice agent logs to create metric variables."""

    @staticmethod
    def _compute_per_turn_latency(context: "_ProcessorContext") -> None:
        """Compute per-turn latency from audio timestamps and save to context.

        Latency is measured as the time from the end of the user's last audio
        segment to the start of the assistant's first audio segment for each turn.
        Turns with missing timestamps are silently skipped.
        """
        latencies: dict[int, float] = {}
        for turn_id, u in context.audio_timestamps_user_turns.items():
            a = context.audio_timestamps_assistant_turns.get(turn_id)
            if not u or not a:
                continue
            latencies[turn_id] = round(a[0][0] - u[-1][1], 6)
        context.latency_assistant_turns = latencies

    def process_record(
        self,
        result: ConversationResult,
        output_dir: Path,
        pipeline_type: PipelineType = PipelineType.CASCADE,
    ) -> _ProcessorContext | None:
        """Process a single conversation record to create metric context.

        Args:
            result: ConversationResult object
            output_dir: Path to the output directory containing logs
            pipeline_type: The type of voice pipeline used

        Returns:
            _ProcessorContext object with all processed variables, or None if processing failed
        """
        context = _ProcessorContext()
        context.record_id = result.record_id
        context.audio_assistant_path = _resolve_path(result.audio_assistant_path, output_dir)
        context.audio_user_path = _resolve_path(result.audio_user_path, output_dir)
        context.audio_mixed_path = _resolve_path(result.audio_mixed_path, output_dir)
        context.pipeline_type = pipeline_type
        context.conversation_ended_reason = result.conversation_ended_reason

        pipecat_path = _resolve_path(result.pipecat_logs_path, output_dir)
        elevenlabs_path = _resolve_path(result.elevenlabs_logs_path, output_dir)

        try:
            self._build_history(context, output_dir, pipecat_path, elevenlabs_path)
            self._extract_turns_from_history(context)
            self._compute_per_turn_latency(context)
            self._reconcile_transcript_with_tools(context)

            return context

        except Exception as e:
            logger.exception(f"Failed to process record {result.record_id}: {e}")
            return None

    @staticmethod
    def _load_audit_log_transcript(output_dir: Path) -> list[dict]:
        """Load and normalize audit log entries into history format."""
        history = []
        audit_log_path = output_dir / "audit_log.json"
        with open(audit_log_path) as f:
            audit_logs = json.load(f)

        transcript = audit_logs.get("transcript", [])
        if not transcript:
            raise ValueError(f"Empty transcript in {audit_log_path}")

        for entry in transcript:
            history.append(
                {
                    "timestamp_ms": int(entry["timestamp"]),
                    "source": "audit_log",
                    "event_type": entry.get("message_type", "unknown"),
                    "data": entry.get("value", {}),
                }
            )
        return history

    @staticmethod
    def _load_pipecat_logs(pipecat_logs_path: str) -> list[dict]:
        """Load and normalize pipecat log entries into history format."""
        history = []
        raw_pipecat = []
        with open(pipecat_logs_path) as f:
            for line in f:
                raw_pipecat.append(json.loads(line))

        allowed_types = {"turn_start", "turn_end", "tts_text", "llm_response"}
        raw_pipecat = [entry for entry in raw_pipecat if entry.get("type") in allowed_types]

        # Some audio-native models emit llm_response (full text with spaces); some emits tts_text (per-token chunks).
        has_tts_text = any(entry.get("type") == "tts_text" for entry in raw_pipecat)
        if has_tts_text:
            raw_pipecat = [entry for entry in raw_pipecat if entry.get("type") != "llm_response"]

        grouped_pipecat = aggregate_pipecat_logs_by_type(raw_pipecat)
        for entry in grouped_pipecat:
            if (ts := entry.get("start_timestamp")) is None:
                continue
            history.append(
                {
                    "timestamp_ms": int(ts),
                    "source": "pipecat",
                    "event_type": entry.get("type", "unknown"),
                    "data": entry.get("data", {}),
                }
            )
        return history

    @staticmethod
    def _load_elevenlabs_logs(elevenlabs_logs_path: str) -> list[dict]:
        """Load and normalize ElevenLabs log entries into history format."""
        history = []
        raw_elevenlabs = []
        with open(elevenlabs_logs_path) as f:
            for line in f:
                raw_elevenlabs.append(json.loads(line))

        filtered_elevenlabs = filter_empty_responses(raw_elevenlabs)
        for entry in filtered_elevenlabs:
            if (ts := entry.get("timestamp")) is None:
                continue
            event_type = entry.get("type") or entry.get("event_type", "unknown")
            data = {k: v for k, v in entry.items() if k not in ("timestamp", "type", "event_type")}
            history.append(
                {
                    "timestamp_ms": int(ts),
                    "source": "elevenlabs",
                    "event_type": event_type,
                    "data": data,
                }
            )
        return history

    def _build_history(
        self,
        context: _ProcessorContext,
        output_dir: Path,
        pipecat_path: str | None,
        elevenlabs_path: str | None,
    ) -> None:
        """Merge audit log, pipecat, and ElevenLabs logs into a timestamp-sorted context.history.

        Each entry: {timestamp_ms, source, event_type, data}.
        """
        history = self._load_audit_log_transcript(output_dir)
        if context.pipeline_type != PipelineType.S2S and pipecat_path:
            history.extend(self._load_pipecat_logs(pipecat_path))
        if elevenlabs_path:
            history.extend(self._load_elevenlabs_logs(elevenlabs_path))

        history.sort(key=lambda e: e["timestamp_ms"])
        context.history = history

        source_counts = Counter(entry["source"] for entry in history)
        logger.info(f"Record {context.record_id}: Built history with {len(history)} events ({dict(source_counts)})")

    @staticmethod
    def _extract_turns_from_history(context: _ProcessorContext) -> None:
        """Extract all turn variables from context.history in a single pass.

        Turn boundaries are driven by ElevenLabs audio_start(elevenlabs_user) events via advance_turn_if_needed().
        Turn 0 = assistant greeting. Index *i* aligns assistant[i] as the reply to user[i].

        Source → variable mapping:
            audit_log/user              → ``transcribed_user_turns[N]``
            pipecat tts_text/llm_response → ``intended_assistant_turns[N]``
            elevenlabs assistant_speech  → ``transcribed_assistant_turns[N]``
            elevenlabs user_speech       → ``intended_user_turns[N]``
            elevenlabs audio_start/end   → ``audio_timestamps_{role}_turns[N]``

        audio_end events are paired with the most recent audio_start of the same role;
        session_end_ts is used as fallback if no audio_end arrives.

        Audio-native models (S2S, AudioLLM) process raw audio — the audit-log user entries are not trustworthy.
        For audio-native pipelines we source user conversation_trace entries from ElevenLabs user_speech
        (intended) instead of the audit-log (transcribed).
        """
        state = _TurnExtractionState()

        conversation_trace: list[dict] = []
        for event in context.history:
            if event["source"] == "audit_log":
                _handle_audit_log_event(event, state, context, conversation_trace, context.pipeline_type)
            elif event["source"] == "pipecat":
                _handle_pipecat_event(event, state, context, conversation_trace)
            elif event["source"] == "elevenlabs":
                if _handle_elevenlabs_event(event, state, context, conversation_trace, context.pipeline_type):
                    continue

        if not state.session_end_ts:
            state.session_end_ts = context.history[-1].get("timestamp_ms") / 1000.0

        _pair_audio_segments(state, context)
        if context.pipeline_type == PipelineType.S2S:
            # S2S has no pipecat segments to validate against — trace entries come from EL directly
            validated_trace = conversation_trace
        else:
            validated_trace = _validate_conversation_trace(conversation_trace, context)
        context.conversation_trace = group_consecutive_turns(validated_trace)
        _fix_interruption_labels(context, state)
        _finalize_extraction(context, state, conversation_trace)

    @staticmethod
    def _reconcile_transcript_with_tools(context: _ProcessorContext) -> None:
        """Reconcile conversation_trace with voice log data.

        - Ensure the assistant greeting (turn 0) is the first trace entry.
        - Append the final user turn if it arrived after the last audit-log entry.
        - Label a trailing assistant turn with CUT_OFF_ON_ITS_OWN and sync it
          across trace / intended / transcribed dicts.
        - Backfill transcribed_user_turns from intended if STT didn't finish.
        """
        if not context.conversation_trace:
            # Empty trace (e.g. greeting-only conversation with no user turns). Create from pipecat intended text if
            # available.
            if context.intended_assistant_turns.get(0):
                context.conversation_trace.append(
                    {
                        "role": "assistant",
                        "content": context.intended_assistant_turns[0],
                        "type": "intended",
                        "turn_id": 0,
                    }
                )
            return

        _ensure_greeting_is_first(context)

        if not context.intended_user_turns:
            return

        last_user_turn_id = max(context.intended_user_turns.keys())
        last_entry = context.conversation_trace[-1]
        last_turn_id = last_entry.get("turn_id")

        # User's final turn arrived after the last audit-log entry — append it and we're done.
        if last_user_turn_id > last_turn_id:
            last_user_text = context.intended_user_turns[last_user_turn_id]
            context.conversation_trace.append(
                {"role": "user", "content": last_user_text, "type": "intended", "turn_id": last_user_turn_id}
            )
            if not context.transcribed_user_turns.get(last_user_turn_id):
                context.transcribed_user_turns[last_user_turn_id] = last_user_text
            logger.info(f"Record {context.record_id}: Appended last user turn: {last_user_text[:50]}")
            return

        _label_trailing_assistant_turn(context, last_entry, last_turn_id)

        # Backfill: if the last intended user turn has no transcription (conversation ended before STT finished), use
        # the intended text.
        if last_user_turn_id is not None and not context.transcribed_user_turns.get(last_user_turn_id):
            last_user_text = context.intended_user_turns[last_user_turn_id]
            context.transcribed_user_turns[last_user_turn_id] = last_user_text
            logger.info(
                f"Record {context.record_id}: Backfilled transcribed_user_turns[{last_user_turn_id}] "
                f"from intended: {last_user_text[:50]}"
            )
