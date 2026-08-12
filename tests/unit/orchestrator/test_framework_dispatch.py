"""Assistant framework dispatch: backend-only via the BackendFactory.

The assistant runs on the Role/Backend path exclusively — the legacy
``_get_server_class`` wiring is gone. A framework the factory backs is usable;
anything else is unported (its legacy server survives only as reference) and the
factory returns ``None`` for it.
"""

from eva.backend.elevenlabs import ElevenLabsBackend
from eva.backend.factory import BackendFactory
from eva.backend.grok_voice import GrokVoiceBackend
from eva.backend.openai_realtime import OpenAIRealtimeBackend

_PORTED = {
    "openai_realtime": OpenAIRealtimeBackend,
    "grok_voice": GrokVoiceBackend,
    "elevenlabs": ElevenLabsBackend,
}

_MINIMAL_CONFIG = {
    "openai_realtime": {"model": "gpt-realtime", "api_key": "k"},
    "grok_voice": {"model": "grok-voice", "api_key": "k"},
    "elevenlabs": {"api_key": "k", "speaker_id": "ag_1"},
}


def test_create_builds_each_ported_backend():
    factory = BackendFactory()
    for name, cls in _PORTED.items():
        assert isinstance(factory.create(name, _MINIMAL_CONFIG[name]), cls)


def test_create_returns_none_for_unported_frameworks():
    # pipecat cascade and gemini_live are not yet native backends -> unusable as assistant.
    factory = BackendFactory()
    assert factory.create("pipecat", {}) is None
    assert factory.create("gemini_live", {}) is None
