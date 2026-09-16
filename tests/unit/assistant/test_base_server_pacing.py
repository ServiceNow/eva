import inspect

from eva.assistant.base_server import AbstractAssistantServer


def test_paced_output_defaults_to_true():
    # The existing ElevenLabs caller depends on real-time cadence for its
    # silence heuristics, so the default must not change.
    signature = inspect.signature(AbstractAssistantServer.__init__)
    assert signature.parameters["paced_output"].default is True


def test_every_server_accepts_paced_output():
    # worker.py always passes paced_output; a server that omits it from its signature
    # raises TypeError at construction (this is what broke the pipecat run).
    import inspect

    from eva.assistant.elevenlabs_server import ElevenLabsAssistantServer
    from eva.assistant.gemini_live_server import GeminiLiveAssistantServer
    from eva.assistant.pipecat_server import PipecatAssistantServer
    from eva.assistant.smallest_hydra_server import SmallestHydraAssistantServer

    for server in (
        PipecatAssistantServer,
        GeminiLiveAssistantServer,
        ElevenLabsAssistantServer,
        SmallestHydraAssistantServer,
    ):
        params = inspect.signature(server.__init__).parameters
        assert "paced_output" in params, f"{server.__name__} rejects paced_output"


def test_only_servers_owning_their_throttle_claim_unpaced_support():
    from eva.assistant.elevenlabs_server import ElevenLabsAssistantServer
    from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
    from eva.assistant.pipecat_server import PipecatAssistantServer

    assert OpenAIRealtimeAssistantServer.supports_unpaced_output is True
    assert PipecatAssistantServer.supports_unpaced_output is False
    assert ElevenLabsAssistantServer.supports_unpaced_output is False
