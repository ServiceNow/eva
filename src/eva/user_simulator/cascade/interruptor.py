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
is speaking right now; their line after the conversation so far is still in progress. Polite callers \
almost never cut an agent off: they let the agent finish, even when they already know what they will \
say. Keep listening unless the agent is clearly getting something about you or your request wrong. \
What should the caller do at this moment?"""

DECISION_CHOICES = [
    {
        "value": "interrupt",
        "description": "Rare. Only when the agent has just stated something clearly wrong about the caller or "
        "their request (a wrong name, number, account or problem) and continuing would waste the call. Never "
        "to answer a question, agree, add information, or because the agent paused or seems to have finished.",
    },
    {
        "value": "keep_listening",
        "description": "The default, almost always: let the agent finish their line, including when they are "
        "asking a question, giving instructions or results, or when the caller already knows their answer.",
    },
]


class DecisionsInterruptor:
    """Asks whether the caller should cut in on the agent's in-progress line."""

    def __init__(self, model: str) -> None:
        self.model = model
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
                        "choices": DECISION_CHOICES,
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
