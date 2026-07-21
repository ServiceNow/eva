"""Tests for the shared conv_finish failure classifier (pure logic over signals)."""

import pytest

from eva.metrics.diagnostic.conv_finish_classifier import (
    ConvFinishSignals,
    classify_conv_finish_failure,
)


def sig(**overrides) -> ConvFinishSignals:
    """A clean parent-failure baseline (fires nothing → unknown_reason), overridable."""
    return ConvFinishSignals(**overrides)


def cat(**overrides) -> str | None:
    return classify_conv_finish_failure(sig(**overrides)).category


# --- not applicable ---------------------------------------------------------
def test_not_parent_failure_returns_none():
    assert classify_conv_finish_failure(sig(is_parent_failure=False)).category is None


def test_clean_parent_failure_is_unknown():
    assert cat() == "unknown_reason"


# --- #1 answer_lost_in_reasoning -------------------------------------------
def test_answer_lost_in_reasoning():
    assert cat(last_perf_response_empty=True, last_perf_reasoning="Great, I booked it.") == ("answer_lost_in_reasoning")


def test_empty_response_but_no_reasoning_is_not_answer_lost():
    # genuine empty generation, not answer-lost
    assert cat(last_perf_response_empty=True, last_perf_reasoning="") == "unknown_reason"


def test_empty_response_with_tool_call_is_not_answer_lost():
    assert cat(last_perf_response_empty=True, last_perf_reasoning="x", last_perf_has_tool_call=True) == (
        "unknown_reason"
    )


# --- #2 / #3 stt no-transcription -------------------------------------------
def test_stt_empty_transcription():
    assert (
        cat(
            is_cascade=True,
            last_conv_message_role="assistant",
            vad_no_tx_warning_after_last_response=True,
            empty_transcript_after_last_response=True,
        )
        == "stt_empty_transcription"
    )


def test_stt_missing_transcription():
    assert (
        cat(
            is_cascade=True,
            last_conv_message_role="assistant",
            vad_no_tx_warning_after_last_response=True,
            empty_transcript_after_last_response=False,
        )
        == "stt_missing_transcription"
    )


def test_stt_needs_cascade():
    assert (
        cat(
            is_cascade=False,
            last_conv_message_role="assistant",
            vad_no_tx_warning_after_last_response=True,
            empty_transcript_after_last_response=True,
        )
        == "unknown_reason"
    )


def test_stt_needs_assistant_last_role():
    # final user turn reached the LLM (role=user) → not an STT drop
    assert (
        cat(
            is_cascade=True,
            last_conv_message_role="user",
            vad_no_tx_warning_after_last_response=True,
            empty_transcript_after_last_response=True,
        )
        == "unknown_reason"
    )


# --- #4 ended_with_user_interruption ---------------------------------------
def test_ended_with_user_interruption():
    assert (
        cat(
            num_interruptions=1,
            accepted_turn_before_last_interruption="the code is 610311",
            response_after_last_interruption=False,
        )
        == "ended_with_user_interruption"
    )


def test_interruption_but_agent_recovered_is_not_it():
    assert (
        cat(
            num_interruptions=1,
            accepted_turn_before_last_interruption="x",
            response_after_last_interruption=True,
        )
        == "unknown_reason"
    )


# --- #5 llm_api_error -------------------------------------------------------
def test_llm_api_error():
    assert cat(llm_api_error_terminal=True) == "llm_api_error"


def test_llm_api_error_outranks_interruption():
    # rec 42 shape: has interruption-like state AND a terminal API error → API error wins
    assert (
        cat(
            llm_api_error_terminal=True,
            num_interruptions=1,
            accepted_turn_before_last_interruption="x",
            response_after_last_interruption=False,
        )
        == "llm_api_error"
    )


# --- #7 vad_no_turn_detected ------------------------------------------------
def test_vad_no_turn_detected():
    assert (
        cat(
            user_final_utterance_after_agent=True,
            vad_onset_in_final_window=False,
            last_conv_message_role="assistant",
        )
        == "vad_no_turn_detected"
    )


def test_vad_fired_is_not_vad_no_turn():
    assert (
        cat(
            user_final_utterance_after_agent=True,
            vad_onset_in_final_window=True,
            last_conv_message_role="assistant",
        )
        == "unknown_reason"
    )


# --- #8 / #9 service errors (infra) — highest priority ----------------------
def test_tts_api_error():
    assert cat(tts_service_error=True) == "tts_api_error"


def test_stt_api_error():
    assert cat(stt_service_error=True) == "stt_api_error"


def test_tts_error_outranks_everything():
    assert (
        cat(
            tts_service_error=True,
            user_final_utterance_after_agent=True,
            vad_onset_in_final_window=False,
            num_interruptions=1,
            accepted_turn_before_last_interruption="x",
            response_after_last_interruption=False,
        )
        == "tts_api_error"
    )


# --- details carried ---------------------------------------------------------
def test_details_include_evidence_for_answer_lost():
    r = classify_conv_finish_failure(sig(last_perf_response_empty=True, last_perf_reasoning="Booked room 201."))
    assert r.category == "answer_lost_in_reasoning"
    assert r.details.get("recovered_answer", "").startswith("Booked room 201")


def test_infra_details_flag_invalid_run():
    r = classify_conv_finish_failure(sig(tts_service_error=True, service_error_name="DeepgramTTSService"))
    assert r.details.get("invalid_run") is True
    assert r.details.get("component") == "tts"


# --- final-turn input flags (orthogonal to the cause classifier) ------------
from eva.metrics.diagnostic.conv_finish_classifier import final_turn_input_flags  # noqa: E402


@pytest.mark.parametrize(
    "text,short,acknowledgement,spelled",
    [
        ("Yes.", True, True, False),
        ("Yes, that is correct.", False, True, False),
        ("Ok thanks", True, True, False),
        ("Correct.", True, True, False),
        ("Confirm.", True, False, False),  # EVABench scope excludes "confirm"
        ("I need a replacement laptop", False, False, False),
        ("No, that is wrong.", False, True, False),  # EVABench confirmation-start includes "no"
        ("floor two", True, False, False),
        ("", False, False, False),
        (None, False, False, False),
        # annotation tags are stripped before short/acknowledgement (EVABench parity)
        ("[neutral] Okay.", True, True, False),
        ("[slightly impatient] Okay.", True, True, False),
        # spelled entities — single-letter / spoken-digit runs, NATO, "as in", caps codes
        ("It is D I A G dash two X M nine five P L Q.", False, False, True),
        ("My employee ID is E M P eight nine seven three zero five", False, False, True),
        ("It is four nine one seven.", False, False, True),
        ("Room two zero one on the second floor.", False, False, True),
        ("The clearance code is C L R dash O C C dash nine five.", False, False, True),
        ("The state abbreviation is G A.", False, False, False),  # only 2 single letters → not a run (≥3)
        ("My confirmation is Alpha Bravo Charlie.", False, False, True),  # NATO
        ("My ID is EMP358.", False, False, True),  # caps alnum code
        ("V as in Victor.", False, False, True),  # phonetic "as in"
        # length caps: long sentences that merely start with an acknowledgement / contain a spell fragment
        ("Yes, that is correct, September fourteenth at three o clock in the afternoon.", False, False, False),
        ("I was calling about my order and my confirmation number is four nine one seven please.", False, False, False),
    ],
)
def test_final_turn_input_flags(text, short, acknowledgement, spelled):
    flags = final_turn_input_flags(text)
    assert flags["short"] is short
    assert flags["acknowledgement"] is acknowledgement
    assert flags["spelled_entity"] is spelled
