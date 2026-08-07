from eva.utils.prompt_manager import PromptManager


def test_cascade_reuses_the_shared_end_call_description():
    # A cascade-specific copy would drift from the other providers' hang-up rules.
    from eva.user_simulator.cascade.simulator import END_CALL_DESCRIPTION as cascade_description
    from eva.user_simulator.openai_realtime import END_CALL_DESCRIPTION as shared_description

    assert cascade_description is shared_description


def test_no_cascade_specific_prompts_remain_in_the_prompt_file():
    # The per-domain user_simulator prompt already carries persona, goal and end_call rules;
    # layering a cascade-only contract on top is what suppressed the end_call tool call.
    from pathlib import Path

    assert "cascade_" not in Path("configs/prompts/simulation.yaml").read_text()


def test_interruption_decision_prompt_has_a_history_slot_and_binary_contract():
    prompt = PromptManager().get_prompt("user_simulator.interruption_decision", conversation_history="AGENT: hello")

    assert "AGENT: hello" in prompt
    assert "YES" in prompt
    assert "NO" in prompt


def test_backchannel_decision_prompt_has_a_history_slot_and_frequency_guidance():
    prompt = PromptManager().get_prompt("user_simulator.backchannel_decision", conversation_history="AGENT: hello")

    assert "AGENT: hello" in prompt
    assert "CURRENTLY SPEAKING, INCOMPLETE" in prompt
    assert "When in doubt, say NO" in prompt
