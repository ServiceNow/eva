"""Barge-in decisions for the cascade caller, over the OpenAI Decisions API."""

from __future__ import annotations

import asyncio
import os

import httpx

OPENAI_API_BASE = "https://api.openai.com/v1"
DECISIONS_PATH = "/decisions"
"""Not a LiteLLM route, so the request goes straight to OpenAI with OPENAI_API_KEY."""

KEEPALIVE = httpx.Limits(max_keepalive_connections=8, keepalive_expiry=300.0)
"""A dedicated pool: decisions must not queue behind reply streams or open a TLS handshake (~100ms)."""

DECISION_INSTRUCTIONS = """You are the caller on a live phone call with a support agent, and the agent \
is speaking right now; their line after the conversation so far is still in progress. Real callers do \
not sit in silence: they cut in when the conversation is obviously going wrong, and they murmur along \
when the agent talks for a long time and is on track for what they want. Otherwise they listen, \
especially early in a line.

What you read is a live speech-to-text transcript of the agent, not their words: names, codes and \
numbers are often misspelled or split oddly (a name heard as a similar-sounding word, "B V" for "B Z"), \
and the last few words may still change. Tell the two apart: a spelling or sound that is close to what you \
know (one or two letters or digits off, a similar-sounding name) is the transcript, so ignore it. A value \
that is plainly a different thing is the agent being wrong: a different flight, date, amount, account, \
name or request, a code that shares almost nothing with yours, or a detail you never gave and contradict.

Do not cut in on a code, name, number or list while it is still being read out, or partway through a \
sentence that has not yet said what it is getting at: let it finish, then judge what they actually said. \
What should the caller do at this moment?"""

DECISION_CHOICES = [
    {
        "value": "interrupt",
        "description": "Something is obviously wrong, whenever in the line it shows: the conversation is heading "
        "away from the caller's goal, or the agent has clearly shown it is using wrong information (a different "
        "flight, date, amount, account, name or problem, a code or number that is plainly not theirs rather "
        "than a near-miss that sounds the same, or a detail the caller never gave). Wait for the clause to "
        "end, not for the whole line. Never to answer a question, agree, add information, summarize, or ask "
        "for something, and never because the agent paused or seems to have finished.",
    },
    {
        "value": "keep_listening",
        "description": "Neutral, and the choice early in a line: nothing is clearly wrong and the line has not "
        "yet gone on long. Includes short or ordinary lines, questions, instructions, details still being read "
        "out or spelled, a possible mismatch that could just be the transcript, or when the caller already "
        "knows their answer.",
    },
]

BACK_CHANNEL_CHOICE = {
    "value": "back_channel",
    "description": 'A brief acknowledgement ("mhm", "yeah", "okay") the way a real listener shows they are '
    "following. Use it when the line has grown long (several sentences, or a long run of details, "
    "instructions, explanations or results) AND what the agent is saying fits the caller's goal and is "
    "heading the right way. It should be the usual choice in a long line like that, over keep_listening. "
    "Never when the agent asked a question or something looks wrong, mid-code or mid-spelling, and not in "
    "a short line.",
}
"""Offered only when back-channeling is enabled."""


class DecisionsInterruptor:
    """Asks whether the caller should cut in on, or murmur along to, the agent's in-progress line."""

    def __init__(self, model: str, *, interrupt: bool = True, back_channel: bool = False) -> None:
        self.model = model
        self._choices = [c for c in DECISION_CHOICES if interrupt or c["value"] != "interrupt"]
        if back_channel:
            self._choices.append(BACK_CHANNEL_CHOICE)
        self._client = httpx.AsyncClient(
            base_url=OPENAI_API_BASE,
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
            timeout=10.0,
            limits=KEEPALIVE,
        )

    async def decide(self, goal: str, history: list[dict[str, str]], transcript: str) -> tuple[str, float]:
        """Return (choice, P(interrupt)) for the agent's line as heard so far.

        ``history`` is the conversation the caller has had so far, in conversation-truth roles
        ("assistant" is the agent); ``transcript`` is the in-progress line, partial included.
        """
        conversation = "\n".join(
            f"{'Agent' if turn['role'] == 'assistant' else 'Caller'}: {turn['content']}" for turn in history
        )
        response = await self._client.post(
            DECISIONS_PATH,
            json={
                "model": self.model,
                "input": f"<user_goal>\n{goal}\n</user_goal>\n\n<conversation>\n{conversation}\n</conversation>"
                f"\n\nAgent (speaking now): {transcript}",
                "questions": [
                    {
                        "type": "choice",
                        "name": "action",
                        "instructions": DECISION_INSTRUCTIONS,
                        "choices": self._choices,
                    }
                ],
            },
        )
        response.raise_for_status()
        answer = response.json()["answers"][0]
        if answer.get("type") == "refusal":
            raise RuntimeError(f"Decisions API refused: {answer}")
        probabilities = {p["value"]: float(p["probability"]) for p in answer.get("probabilities") or []}
        return str(answer.get("choice", "")), probabilities.get("interrupt", 0.0)

    async def warm_up(self, goal: str, *, connections: int = 1) -> None:
        """Open ``connections`` pooled connections now, so no handshake lands on a barge-in."""
        await asyncio.gather(*(self.decide(goal, [], "Hello.") for _ in range(connections)))

    async def close(self) -> None:
        await self._client.aclose()
