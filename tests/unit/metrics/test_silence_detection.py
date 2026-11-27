"""Vectorized silence detection matches pydub, and speech fidelity audio work stays off the event loop."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydub import AudioSegment
from pydub.silence import detect_nonsilent as pydub_detect_nonsilent

from eva.metrics.accuracy.speech_fidelity import SpeechFidelityMetric
from eva.metrics.utils import detect_nonsilent
from eva.metrics.validation.user_speech_fidelity import UserSpeechFidelityMetric
from eva.models.config import PipelineType

from .conftest import make_judge_metric, make_metric_context
from .test_agent_speech_fidelity import SIMPLE_TRACE


def _segment(samples: np.ndarray, frame_rate: int, channels: int = 1) -> AudioSegment:
    return AudioSegment(
        np.clip(samples, -32768, 32767).astype("<i2").tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=channels,
    )


def _speech_and_silence(rng: np.random.Generator, frame_rate: int, seconds: float, channels: int) -> np.ndarray:
    """Bursts of loud noise separated by digital silence, faint noise near -90 dBFS, or both."""
    n = int(seconds * frame_rate) * channels
    out = np.zeros(n, dtype=np.int64)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.05, 4.5) * frame_rate) * channels
        kind = rng.integers(0, 3)
        if kind == 0:
            out[pos : pos + length] = rng.normal(0, 4000, size=min(length, n - pos))
        elif kind == 1:
            out[pos : pos + length] = rng.integers(-2, 3, size=min(length, n - pos))  # rms right around 1
        pos += length
    return out


CASES = [
    # (frame_rate, channels, seconds, min_silence_len, silence_thresh, seek_step)
    (24000, 1, 40.0, 3000, -90, 50),  # production settings
    (24000, 1, 40.0137, 3000, -90, 50),  # length not a whole number of ms, last slice not on a seek step
    (16000, 1, 30.0, 3000, -40, 50),
    (8000, 1, 12.345, 500, -60, 7),
    (44100, 2, 10.0, 1000, -50, 10),  # stereo, fractional frames per ms
    (22050, 1, 8.0, 700, -90, 1),
]


@pytest.mark.parametrize(("frame_rate", "channels", "seconds", "min_silence_len", "silence_thresh", "seek_step"), CASES)
@pytest.mark.parametrize("seed", range(4))
def test_matches_pydub(frame_rate, channels, seconds, min_silence_len, silence_thresh, seek_step, seed):
    rng = np.random.default_rng(seed)
    seg = _segment(_speech_and_silence(rng, frame_rate, seconds, channels), frame_rate, channels)
    expected = pydub_detect_nonsilent(seg, min_silence_len, silence_thresh, seek_step)
    assert detect_nonsilent(seg, min_silence_len, silence_thresh, seek_step) == expected


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param(np.zeros(24000 * 10), id="all-silent"),
        pytest.param(np.full(24000 * 10, 5000), id="all-loud"),
        pytest.param(np.full(24000 * 2, 5000), id="shorter-than-min-silence"),
        pytest.param(np.concatenate([np.zeros(24000 * 4), np.full(24000 * 4, 5000)]), id="leading-silence"),
        pytest.param(np.concatenate([np.full(24000 * 4, 5000), np.zeros(24000 * 4)]), id="trailing-silence"),
        pytest.param(np.zeros(0), id="empty"),
    ],
)
def test_edge_cases_match_pydub(samples):
    seg = _segment(samples, 24000)
    assert detect_nonsilent(seg, 3000, -90, 50) == pydub_detect_nonsilent(seg, 3000, -90, 50)


@pytest.mark.parametrize("metric_cls", [SpeechFidelityMetric, UserSpeechFidelityMetric])
async def test_audio_work_runs_off_the_event_loop(metric_cls):
    """Load, trim and encode must not run on the loop thread, which live conversations share."""
    metric = make_judge_metric(metric_cls, mock_llm=True)
    metric.llm_client.generate_text.return_value = (None, None)
    loop_thread = threading.get_ident()
    threads: dict[str, int] = {}

    def record(name, result):
        def fn(*_args, **_kwargs):
            threads[name] = threading.get_ident()
            return result

        return fn

    metric.load_role_audio = record("load", MagicMock())
    metric._trim_silence = record("trim", MagicMock())
    metric.encode_audio_segment = record("encode", "base64audio")
    context = make_metric_context(
        audio_assistant_path="/fake/audio_assistant.wav",
        audio_user_path="/fake/audio_user.wav",
        pipeline_type=PipelineType.S2S,
        conversation_trace=SIMPLE_TRACE,
    )

    await metric.compute(context)

    assert set(threads) == {"load", "trim", "encode"}
    assert loop_thread not in threads.values()
