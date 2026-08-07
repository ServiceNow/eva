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
