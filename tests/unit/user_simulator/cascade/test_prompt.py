def test_cascade_reuses_the_shared_end_call_description():
    # A cascade-specific copy would drift from the other providers' hang-up rules.
    from eva.user_simulator.cascade.simulator import END_CALL_DESCRIPTION as cascade_description
    from eva.user_simulator.openai_realtime import END_CALL_DESCRIPTION as shared_description

    assert cascade_description is shared_description


def test_the_turn_call_carries_no_cascade_specific_contract():
    # The per-domain user_simulator prompt already carries persona, goal and end_call rules;
    # layering a cascade-only contract on top is what suppressed the end_call tool call.
    from eva.user_simulator.cascade.simulator import CascadeUserSimulator

    sim = object.__new__(CascadeUserSimulator)
    sim._build_prompt = lambda: "SYSTEM PROMPT"
    sim._history = []

    assert sim._messages()[0]["content"] == "SYSTEM PROMPT"
