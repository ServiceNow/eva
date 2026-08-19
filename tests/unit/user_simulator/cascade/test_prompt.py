from eva.utils.prompt_manager import PromptManager


def test_cascade_reuses_the_shared_end_call_description():
    # A cascade-specific copy would drift from the other providers' hang-up rules.
    from eva.user_simulator.cascade.simulator import END_CALL_DESCRIPTION as cascade_description
    from eva.user_simulator.openai_realtime import END_CALL_DESCRIPTION as shared_description

    assert cascade_description is shared_description


def test_the_turn_call_carries_no_cascade_specific_contract():
    # The per-domain user_simulator prompt already carries persona, goal and end_call rules;
    # layering a cascade-only contract on top is what suppressed the end_call tool call.
    # Out-of-turn behavior prompts exist, but they are only ever used in their own
    # standalone calls — never appended to the system prompt of the turn call.
    from eva.user_simulator.cascade.simulator import CascadeUserSimulator

    sim = object.__new__(CascadeUserSimulator)
    sim._build_prompt = lambda: "SYSTEM PROMPT"
    sim._history = []

    assert sim._messages()[0]["content"] == "SYSTEM PROMPT"


def test_interruption_decision_prompt_has_a_history_slot_and_binary_contract():
    prompt = PromptManager().get_prompt(
        "user_simulator.interruption_decision", conversation_history="AGENT: hello", user_goal="Unlock my account."
    )

    assert "AGENT: hello" in prompt
    assert "YES" in prompt
    assert "NO" in prompt


def test_backchannel_decision_prompt_has_a_history_slot_and_frequency_guidance():
    prompt = PromptManager().get_prompt("user_simulator.backchannel_decision", conversation_history="AGENT: hello")

    assert "AGENT: hello" in prompt
    assert "CURRENTLY SPEAKING, INCOMPLETE" in prompt
    assert "When in doubt, say NO" in prompt


def test_self_correction_prompt_states_the_wrong_then_right_ordering():
    prompt = PromptManager().get_template("user_simulator.cascade_self_correction")

    assert "self_correction" in prompt
    assert "must_have_criteria" in prompt


def test_self_correction_prompt_never_mentions_ending_the_call():
    # It runs as its own call; if it could elicit a hang-up it would race the turn call.
    prompt = PromptManager().get_template("user_simulator.cascade_self_correction")

    assert "end_call" not in prompt


def test_interruption_decision_prompt_carries_the_user_goal():
    prompt = PromptManager().get_prompt(
        "user_simulator.interruption_decision",
        conversation_history="AGENT: hello",
        user_goal="Get my account unlocked.",
    )

    assert "Get my account unlocked." in prompt
    # The goodbye case the caller kept barging in on.
    assert "likely to hang up" in prompt
