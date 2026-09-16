from eva.user_simulator.cascade.decisions import ListenerDecisions, parse_yes_no


def test_parse_yes_no_accepts_plain_yes():
    assert parse_yes_no("YES") is True


def test_parse_yes_no_is_case_and_whitespace_insensitive():
    assert parse_yes_no("  yes\n") is True


def test_parse_yes_no_treats_anything_else_as_no():
    assert parse_yes_no("NO") is False
    assert parse_yes_no("maybe") is False
    assert parse_yes_no("") is False


class FakeLLM:
    """Returns scripted replies, or raises when configured to."""

    def __init__(self, replies: list[str] | None = None, error: Exception | None = None) -> None:
        self.replies = replies or []
        self.error = error
        self.calls = 0

    async def decide(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.replies.pop(0) if self.replies else "NO"


async def test_both_checks_run_and_interrupt_wins_ties():
    llm = FakeLLM(["YES", "YES"])
    decisions = ListenerDecisions(
        llm, interrupt_prompt="i {conversation_history}", backchannel_prompt="b {conversation_history}"
    )

    verdict = await decisions.evaluate("AGENT: hello", allow_interrupt=True, allow_backchannel=True)

    assert verdict.should_interrupt is True
    assert verdict.should_backchannel is False
    assert llm.calls == 2


async def test_backchannel_alone_when_interrupt_declines():
    llm = FakeLLM(["NO", "YES"])
    decisions = ListenerDecisions(
        llm, interrupt_prompt="i {conversation_history}", backchannel_prompt="b {conversation_history}"
    )

    verdict = await decisions.evaluate("AGENT: hello", allow_interrupt=True, allow_backchannel=True)

    assert verdict.should_interrupt is False
    assert verdict.should_backchannel is True


async def test_disabled_behaviors_are_not_called_at_all():
    llm = FakeLLM(["YES"])
    decisions = ListenerDecisions(
        llm, interrupt_prompt="i {conversation_history}", backchannel_prompt="b {conversation_history}"
    )

    verdict = await decisions.evaluate("AGENT: hello", allow_interrupt=False, allow_backchannel=True)

    assert verdict.should_interrupt is False
    assert llm.calls == 1


async def test_a_failing_check_fails_closed():
    llm = FakeLLM(error=RuntimeError("provider down"))
    decisions = ListenerDecisions(
        llm, interrupt_prompt="i {conversation_history}", backchannel_prompt="b {conversation_history}"
    )

    verdict = await decisions.evaluate("AGENT: hello", allow_interrupt=True, allow_backchannel=True)

    assert verdict.should_interrupt is False
    assert verdict.should_backchannel is False


async def test_the_goal_is_substituted_into_the_interrupt_prompt():
    llm = FakeLLM(["NO", "NO"])
    seen: list[str] = []

    class _Recorder(FakeLLM):
        async def decide(self, prompt: str) -> str:
            seen.append(prompt)
            return await super().decide(prompt)

    decisions = ListenerDecisions(
        _Recorder(["NO", "NO"]),
        interrupt_prompt="goal={user_goal} history={conversation_history}",
        backchannel_prompt="b {conversation_history}",
        user_goal="Unlock my account.",
    )

    await decisions.evaluate("AGENT: hello", allow_interrupt=True, allow_backchannel=True)

    assert "goal=Unlock my account." in seen[0]
    assert llm.calls == 0


async def test_a_backchannel_prompt_without_a_goal_slot_still_works():
    # str.format ignores unused keyword arguments, so one signature serves both prompts.
    decisions = ListenerDecisions(
        FakeLLM(["NO", "YES"]),
        interrupt_prompt="i {conversation_history}",
        backchannel_prompt="b {conversation_history}",
        user_goal="Unlock my account.",
    )

    verdict = await decisions.evaluate("AGENT: hello", allow_interrupt=True, allow_backchannel=True)

    assert verdict.should_backchannel is True


def test_goal_summary_is_compact_and_names_what_is_left():
    from eva.user_simulator.cascade.simulator import summarize_goal

    summary = summarize_goal(
        {
            "high_level_user_goal": "Unlock my AD account.",
            "decision_tree": {
                "must_have_criteria": ["account unlocked"],
                "nice_to_have_criteria": ["a case number"],
                "negotiation_behavior": "take the fastest fix offered",
                "resolution_condition": "user can sign in",
                "failure_condition": "agent cannot unlock it",
                "escalation_behavior": "do not ask for a live agent",
                "edge_cases": ["a very long irrelevant edge case " * 40],
            },
        }
    )

    for expected in (
        "Unlock my AD account.",
        "account unlocked",
        "a case number",
        "take the fastest fix offered",
        "user can sign in",
        "agent cannot unlock it",
        "do not ask for a live agent",
    ):
        assert expected in summary, expected
    # edge_cases is long and describes how to answer questions, not whether the goal is done.
    assert "irrelevant" not in summary


def test_goal_summary_omits_absent_fields_without_blank_labels():
    from eva.user_simulator.cascade.simulator import summarize_goal

    summary = summarize_goal({"high_level_user_goal": "Unlock my account.", "decision_tree": {}})

    assert summary == "GOAL: Unlock my account."
