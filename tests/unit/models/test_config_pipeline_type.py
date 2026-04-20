"""Tests for PipelineType enum and get_pipeline_type helper."""

from eva.models.config import PipelineType, get_pipeline_type


class TestGetPipelineType:
    def test_cascade_from_llm_key(self):
        assert get_pipeline_type({"llm": "gpt-4o"}) == PipelineType.CASCADE

    def test_s2s_from_s2s_key(self):
        assert get_pipeline_type({"s2s": "gpt-realtime-mini"}) == PipelineType.S2S

    def test_audio_llm_from_audio_llm_key(self):
        assert get_pipeline_type({"audio_llm": "ultravox"}) == PipelineType.AUDIO_LLM

    def test_legacy_realtime_model_returns_s2s(self):
        assert get_pipeline_type({"realtime_model": "gpt-4o-realtime"}) == PipelineType.S2S

    def test_empty_dict_returns_cascade(self):
        assert get_pipeline_type({}) == PipelineType.CASCADE
