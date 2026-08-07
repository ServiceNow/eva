from eva.utils.prompt_manager import PromptManager


def test_cascade_turn_contract_prompt_exists_and_names_the_json_field():
    prompt = PromptManager().get_prompt("user_simulator.cascade_turn_contract")

    assert "utterance" in prompt
    assert "JSON" in prompt
    assert "end_call" in prompt
