"""Apriel routing must not replace other providers' adapters."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from eva.assistant.apriel_realtime_server import AprielRealtimeAssistantServer
from eva.assistant.grok_voice_server import GrokVoiceAssistantServer
from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
from eva.orchestrator.worker import ConversationWorker, _get_server_class


@pytest.mark.parametrize("provider", ["apriel-realtime", "APRIEL-REALTIME"])
def test_explicit_apriel_provider_uses_dedicated_adapter(provider):
    assert _get_server_class("openai_realtime", provider) is AprielRealtimeAssistantServer


@pytest.mark.parametrize("provider", [None, "", "openai", "gpt-realtime", "custom-openai", "custom-provider"])
def test_other_openai_providers_keep_original_adapter(provider):
    assert _get_server_class("openai_realtime", provider) is OpenAIRealtimeAssistantServer


def test_existing_single_argument_dispatch_keeps_original_adapter():
    assert _get_server_class("openai_realtime") is OpenAIRealtimeAssistantServer


@pytest.mark.parametrize("provider", [None, "grok", "apriel-realtime"])
def test_other_framework_is_not_redirected(provider):
    assert _get_server_class("grok_voice", provider) is GrokVoiceAssistantServer
    assert issubclass(GrokVoiceAssistantServer, OpenAIRealtimeAssistantServer)
    assert not issubclass(GrokVoiceAssistantServer, AprielRealtimeAssistantServer)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "apriel-realtime"])
async def test_worker_passes_provider_to_dispatch(monkeypatch, provider):
    server = SimpleNamespace(start=AsyncMock())
    server_class = Mock(return_value=server)
    dispatch = Mock(return_value=server_class)
    monkeypatch.setattr("eva.orchestrator.worker._get_server_class", dispatch)
    worker = object.__new__(ConversationWorker)
    worker.config = SimpleNamespace(framework="openai_realtime", model=SimpleNamespace(s2s=provider), language="en")
    worker.record = SimpleNamespace(current_date_time="2026-01-01T00:00:00", id="test")
    worker._materialize_resolved_scenario_db = Mock(return_value=Path("scenario.json"))
    worker.agent = Mock()
    worker.agent_config_path = "agent.yaml"
    worker.output_dir = Path("output")
    worker.port = 9000

    await worker._start_assistant()

    dispatch.assert_called_once_with("openai_realtime", provider)
    assert server_class.call_args.kwargs["pipeline_config"] is worker.config.model
    server.start.assert_awaited_once()


def test_apriel_keeps_existing_eva_event_parser():
    assert AprielRealtimeAssistantServer._metrics_processor_name == "openai_realtime"
