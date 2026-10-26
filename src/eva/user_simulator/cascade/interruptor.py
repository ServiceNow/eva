"""Barge-in decisions for the cascade caller, over the OpenAI Decisions API."""

from __future__ import annotations

import asyncio
import os

import httpx

from eva.utils import router

INTERRUPTOR_DEPLOYMENT = "interruptor-llm"
"""EVA_MODEL_LIST deployment whose litellm_params (model, api_key, api_base) the decisions are sent with."""

DECISIONS_PATH = "/decisions"
"""Not a LiteLLM route, so the request goes straight to the deployment's API base."""

KEEPALIVE = httpx.Limits(max_keepalive_connections=8, keepalive_expiry=300.0)
"""A dedicated pool: decisions must not queue behind reply streams or open a TLS handshake (~100ms)."""

DECISION_INSTRUCTIONS = """You are the caller on a live phone call with a support agent, and the agent \
is speaking right now; the final line of the transcript is still in progress. What should the \
caller do at this moment?"""

DECISION_CHOICES = [
    {
        "value": "interrupt",
        "description": "Cut in now: the caller has heard enough to have a response, question or "
        "correction ready, needs to redirect the agent, or already has what they were waiting for.",
    },
    {
        "value": "keep_listening",
        "description": "Let the agent continue: they have said only a few words, or the caller would only be agreeing.",
    },
]


def _deployment_params(model_name: str) -> dict:
    for deployment in getattr(router.get(), "model_list", []):
        if deployment.get("model_name") == model_name:
            return deployment.get("litellm_params", {})
    raise ValueError(f"No {model_name!r} deployment in EVA_MODEL_LIST")


class DecisionsInterruptor:
    """Asks whether the caller should cut in on the agent's in-progress line."""

    def __init__(self, model_name: str = INTERRUPTOR_DEPLOYMENT) -> None:
        params = _deployment_params(model_name)
        self.model = str(params["model"]).split("/", 1)[-1]  # "openai/gpt-6-luna" -> "gpt-6-luna"
        self._client = httpx.AsyncClient(
            base_url=params.get("api_base") or "https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {params.get('api_key') or os.environ.get('OPENAI_API_KEY', '')}"},
            timeout=10.0,
            limits=KEEPALIVE,
        )

    async def decide(self, goal: str, transcript: str) -> tuple[str, float]:
        """Return (choice, P(interrupt)) for the agent's line as heard so far."""
        response = await self._client.post(
            DECISIONS_PATH,
            json={
                "model": self.model,
                "input": f"<user_goal>\n{goal}\n</user_goal>\n\nAgent (speaking now): {transcript}",
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
        await asyncio.gather(*(self.decide(goal, "Hello.") for _ in range(connections)))

    async def close(self) -> None:
        await self._client.aclose()
