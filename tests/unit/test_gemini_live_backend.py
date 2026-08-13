"""Unit tests for the Gemini Live ``Backend`` (no network).

Covers the pure surfaces: config validation (model required, accent rejected,
cumulative errors, optional api_key), sample-rate/capability exposure, tool-schema
translation, the stateful ``LiveServerMessage`` -> ``BackendEvent`` mapping
(audio, transcripts, turn completion, interruption, tool calls, usage), factory
dispatch, and ``send()`` validation. The live session is out of scope here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eva.backend.base import BackendEventType, ToolCallResult
from eva.backend.factory import BackendFactory
from eva.backend.gemini_live import DEFAULT_VOICE, GeminiLiveBackend, GeminiLiveSession


def _backend(**overrides) -> GeminiLiveBackend:
    config = {"model": "gemini-live-2.5-flash", "api_key": "k", **overrides}
    return GeminiLiveBackend(config=config)


def _session() -> GeminiLiveSession:
    return GeminiLiveSession(client=None, conn_cm=None, live=None)  # type: ignore[arg-type]


_BACKEND = _backend()


def _map_response(session, response):
    return _BACKEND._map_response(session, response)


# ── Config validation ────────────────────────────────────────────────────


def test_requires_model():
    with pytest.raises(ValueError, match="model"):
        GeminiLiveBackend(config={"api_key": "k"})


def test_accent_is_rejected():
    with pytest.raises(ValueError, match="accent"):
        GeminiLiveBackend(config={"model": "m", "accent": "british"})


def test_config_errors_are_cumulative():
    with pytest.raises(ValueError) as exc:
        GeminiLiveBackend(config={"accent": "british"})  # missing model AND accent set
    msg = str(exc.value)
    assert "model" in msg and "accent" in msg


def test_api_key_is_optional(monkeypatch):
    # Unlike the OpenAI family, Gemini may auth via Vertex/ADC -> no api_key needed to construct.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    b = GeminiLiveBackend(config={"model": "gemini-live-2.5-flash"})
    assert b._api_key == ""


def test_api_key_falls_back_to_google_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    assert GeminiLiveBackend(config={"model": "m"})._api_key == "env-key"


def test_speaker_id_maps_to_voice_with_default():
    assert _backend()._voice == DEFAULT_VOICE
    assert _backend(speaker_id="Puck")._voice == "Puck"


def test_language_code_precedence():
    assert _backend(language="en")._language_code == "en"
    # explicit language_code wins over the run language
    assert _backend(language="en", language_code="fr-FR")._language_code == "fr-FR"


def test_sample_rates_from_input_format():
    assert _backend().input_sample_rate == 24000  # pcm: role rate
    assert _backend().output_sample_rate == 24000
    assert _backend(input_format="pcmu").input_sample_rate == 8000


def test_capabilities():
    caps = _backend().capabilities
    assert caps.emits_continuous_audio is True
    assert caps.supports_streaming_interruption is True
    assert caps.owns_playout_clock is False


def test_factory_dispatch():
    b = BackendFactory().create("gemini_live", {"model": "m", "api_key": "k"})
    assert isinstance(b, GeminiLiveBackend)


# ── Tool-schema translation ──────────────────────────────────────────────


def test_format_tools_translates_to_gemini_declarations():
    tools = [
        {
            "name": "get_reservation",
            "description": "look up",
            "parameters": {
                "type": "object",
                "properties": {"confirmation_number": {"type": "string", "description": "code"}},
                "required": ["confirmation_number"],
            },
        }
    ]
    (tool,) = GeminiLiveBackend._format_tools(tools)
    (decl,) = tool.function_declarations
    assert decl.name == "get_reservation"
    assert "confirmation_number" in decl.parameters.properties
    assert decl.parameters.required == ["confirmation_number"]


def test_format_tools_none_becomes_none():
    assert GeminiLiveBackend._format_tools(None) is None
    assert GeminiLiveBackend._format_tools([]) is None


# ── Event mapping ────────────────────────────────────────────────────────


def _server_content(**kwargs):
    return SimpleNamespace(server_content=SimpleNamespace(**kwargs), tool_call=None, usage_metadata=None)


def test_model_turn_emits_turn_started_then_audio():
    s = _session()
    part = SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00\x01\x02\x03\x04\x05"))
    events = _map_response(s, _server_content(model_turn=SimpleNamespace(parts=[part])))
    assert [e.event_type for e in events] == [BackendEventType.OUTPUT_TURN_STARTED, BackendEventType.AUDIO_OUTPUT]
    assert events[1].audio == b"\x00\x01\x02\x03\x04\x05"
    assert s.in_model_turn is True
    # A second model_turn chunk does not re-emit OUTPUT_TURN_STARTED.
    events2 = _map_response(s, _server_content(model_turn=SimpleNamespace(parts=[part])))
    assert [e.event_type for e in events2] == [BackendEventType.AUDIO_OUTPUT]


def test_input_transcription_emits_input_transcript():
    (be,) = _map_response(_session(), _server_content(input_transcription=SimpleNamespace(text="hi there")))
    assert be.event_type == BackendEventType.TRANSCRIPT
    assert be.transcript == "hi there"
    assert be.metadata == {"stream": "input", "final": True}


def test_output_transcription_accumulates_then_turn_complete_emits():
    s = _session()
    assert _map_response(s, _server_content(output_transcription=SimpleNamespace(text="Hello"))) == []
    assert _map_response(s, _server_content(output_transcription=SimpleNamespace(text="there"))) == []
    events = _map_response(s, _server_content(turn_complete=True))
    types_ = [e.event_type for e in events]
    assert types_ == [BackendEventType.TRANSCRIPT, BackendEventType.OUTPUT_AUDIO_DONE, BackendEventType.TURN_END]
    assert events[0].transcript == "Hello there"
    assert events[0].metadata == {"stream": "output", "final": True}
    assert events[-1].metadata["interrupted"] is False
    assert s.in_model_turn is False  # reset


def test_interrupted_emits_turn_end_and_resets():
    s = _session()
    _map_response(s, _server_content(output_transcription=SimpleNamespace(text="partial")))
    (be,) = _map_response(s, _server_content(interrupted=True))
    assert be.event_type == BackendEventType.TURN_END
    assert be.transcript == "partial"
    assert be.metadata["interrupted"] is True
    assert s.output_transcript_parts == []


def test_tool_call_emits_request_and_records_name():
    s = _session()
    fc = SimpleNamespace(id="c1", name="get_reservation", args={"confirmation_number": "ABC"})
    msg = SimpleNamespace(server_content=None, tool_call=SimpleNamespace(function_calls=[fc]), usage_metadata=None)
    (be,) = _map_response(s, msg)
    assert be.event_type == BackendEventType.TOOL_CALL_REQUEST
    assert be.tool_call_request.call_id == "c1"
    assert be.tool_call_request.arguments == {"confirmation_number": "ABC"}
    assert s.tool_names["c1"] == "get_reservation"
    assert s.has_function_calls is True


def test_usage_metadata_captured_and_surfaced_on_turn_end():
    s = _session()
    usage_msg = SimpleNamespace(
        server_content=None,
        tool_call=None,
        usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=7),
    )
    _map_response(s, usage_msg)
    assert s.usage == {"prompt_tokens": 11, "completion_tokens": 7}
    events = _map_response(s, _server_content(turn_complete=True))  # [OUTPUT_AUDIO_DONE, TURN_END]
    turn_end = events[-1]
    assert turn_end.event_type == BackendEventType.TURN_END
    assert turn_end.metadata["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}


def test_empty_response_yields_nothing():
    assert _map_response(_session(), SimpleNamespace(server_content=None, tool_call=None, usage_metadata=None)) == []


# ── send() validation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_requires_exactly_one_arg():
    b, s = _backend(), _session()
    with pytest.raises(ValueError):
        await b.send(s)
    with pytest.raises(ValueError):
        await b.send(s, audio=b"x", text="y")


@pytest.mark.asyncio
async def test_send_wrong_session_type_raises():
    with pytest.raises(TypeError):
        await _backend().send(object(), tool_result=ToolCallResult(call_id="c", result={}))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_close_is_idempotent_and_network_free():
    b, s = _backend(), _session()
    await b.close(s)
    await b.close(s)


# ── Client routing (Vertex vs Developer API) ─────────────────────────────
#
# Vertex-only Live/S2S preview models must route through Vertex AI (vertexai=True);
# a Developer API key would 404 them, so Vertex must win whenever a project is
# resolvable (or GOOGLE_GENAI_USE_VERTEXAI is set) and the key is ignored then.


@pytest.fixture
def mock_genai_client():
    with patch("eva.backend.gemini_live.genai.Client") as mock:
        yield mock


class TestCreateClientRouting:
    def test_vertex_when_project_resolvable_ignores_dev_key(self, mock_genai_client, monkeypatch):
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        _backend(project="proj-x", api_key="AIzaSyDEVKEY")._create_client()
        _, kwargs = mock_genai_client.call_args
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "proj-x"
        assert kwargs["location"] == "us-central1"
        assert "api_key" not in kwargs

    def test_use_vertexai_flag_forces_vertex(self, mock_genai_client, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
        _backend(project="proj-x", api_key="AIzaSyDEVKEY")._create_client()
        _, kwargs = mock_genai_client.call_args
        assert kwargs["vertexai"] is True

    def test_flag_zero_forces_dev_api_when_key_present(self, mock_genai_client, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "0")
        _backend(project="proj-x", api_key="AIzaSyDEVKEY")._create_client()
        _, kwargs = mock_genai_client.call_args
        assert kwargs.get("api_key") == "AIzaSyDEVKEY"
        assert "vertexai" not in kwargs

    def test_vertex_flag_without_project_raises(self, mock_genai_client, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        with pytest.raises(ValueError, match="no project found"):
            _backend()._create_client()

    def test_endpoint_and_api_version_passed_via_http_options(self, mock_genai_client, monkeypatch):
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        _backend(
            project="proj-x",
            endpoint="us-central1-aiplatform.googleapis.com",
            api_version="v1beta1",
        )._create_client()
        _, kwargs = mock_genai_client.call_args
        http_options = kwargs["http_options"]
        assert http_options.base_url == "wss://us-central1-aiplatform.googleapis.com"
        assert http_options.api_version == "v1beta1"

    def test_dev_api_when_only_key_and_no_project(self, mock_genai_client, monkeypatch):
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        _backend(api_key="AIzaSyDEVKEY")._create_client()
        _, kwargs = mock_genai_client.call_args
        assert kwargs.get("api_key") == "AIzaSyDEVKEY"
        assert "vertexai" not in kwargs

    def test_global_location_forced_to_region(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
        assert _backend(location="global")._vertex_location == "us-central1"


# ── thinking_config / function_response_scheduling ───────────────────────


def test_thinking_config_defaults_to_empty():
    # No thinking_config -> a default ThinkingConfig is still attached to the live config.
    cfg = _backend()._build_live_config("sys", None)
    assert cfg.thinking_config is not None


def test_thinking_config_parses_flat_dict():
    b = _backend(thinking_config={"thinking_budget": 1024, "include_thoughts": False})
    assert b._thinking_config.thinking_budget == 1024
    assert b._thinking_config.include_thoughts is False


@pytest.mark.asyncio
async def test_scheduling_omitted_by_default(monkeypatch):
    b, s = _backend(), _session()
    captured = {}

    async def fake_send(function_responses):
        captured["fr"] = function_responses[0]

    s.live = SimpleNamespace(send_tool_response=fake_send)
    s.tool_names["c1"] = "get_reservation"
    await b.send(s, tool_result=ToolCallResult(call_id="c1", result={"ok": True}))
    assert captured["fr"].scheduling is None


@pytest.mark.asyncio
async def test_scheduling_set_when_configured():
    b, s = _backend(function_response_scheduling="WHEN_IDLE"), _session()
    captured = {}

    async def fake_send(function_responses):
        captured["fr"] = function_responses[0]

    s.live = SimpleNamespace(send_tool_response=fake_send)
    s.tool_names["c1"] = "get_reservation"
    await b.send(s, tool_result=ToolCallResult(call_id="c1", result={"ok": True}))
    assert captured["fr"].scheduling is not None
