import inspect

from eva.assistant.base_server import AbstractAssistantServer


def test_paced_output_defaults_to_true():
    # The existing ElevenLabs caller depends on real-time cadence for its
    # silence heuristics, so the default must not change.
    signature = inspect.signature(AbstractAssistantServer.__init__)
    assert signature.parameters["paced_output"].default is True
