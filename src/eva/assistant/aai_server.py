"""EVA assistant server for the aai voice-agent framework.

Bridges EVA's user simulator to an aai host-mode session. All the contract
plumbing lives in ``WebSocketBridgeAssistantServer``; this class only builds the
per-session agent definition — system prompt, greeting, and tool schemas — and
opens the backend.

Requires a running aai host with ``AAI_ALLOW_HOST`` enabled. See
docs/aai_integration.md.
"""

import os

from eva.assistant.aai_session import (
    DEFAULT_AAI_INPUT_SAMPLE_RATE,
    DEFAULT_AAI_MODEL,
    DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
    DEFAULT_AAI_WS_URL,
    AAIHostSession,
)
from eva.assistant.bridge_events import VoiceBackendSession
from eva.assistant.ws_bridge_server import WebSocketBridgeAssistantServer
from eva.utils.logging import get_logger

logger = get_logger(__name__)


class AAIAssistantServer(WebSocketBridgeAssistantServer):
    """Runs an EVA conversation against an aai host-mode agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        s2s_params = self.pipeline_config.s2s_params or {}
        self._model: str = s2s_params.get("model") or DEFAULT_AAI_MODEL
        self._ws_url: str = s2s_params.get("ws_url") or os.environ.get("AAI_WS_URL") or DEFAULT_AAI_WS_URL
        self._input_rate: int = int(s2s_params.get("input_sample_rate") or DEFAULT_AAI_INPUT_SAMPLE_RATE)
        self._output_rate: int = int(s2s_params.get("output_sample_rate") or DEFAULT_AAI_OUTPUT_SAMPLE_RATE)
        # Only a deployed agent needs this; a local `aai dev` host takes no key.
        self._api_key: str | None = s2s_params.get("api_key") or os.environ.get("AAI_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model

    def _build_aai_tools(self) -> list[dict]:
        """Convert the agent's tools to aai's flat ToolSchema shape.

        aai validates ``{type, name, description, parameters}`` with a non-empty
        description and a JSON Schema object for parameters.
        """
        tools: list[dict] = []
        for tool in self.agent.tools or []:
            description = f"{tool.name}: {tool.description}".strip(": ").strip() or tool.function_name
            tools.append(
                {
                    "type": "function",
                    "name": tool.function_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.get_parameter_properties(),
                        "required": tool.get_required_param_names(),
                    },
                }
            )
        return tools

    async def _open_backend(self) -> VoiceBackendSession:
        return await AAIHostSession.connect(
            ws_url=self._ws_url,
            system_prompt=self._build_system_prompt(),
            tools=self._build_aai_tools(),
            greeting=self.initial_message,
            input_rate=self._input_rate,
            output_rate=self._output_rate,
            api_key=self._api_key,
        )
