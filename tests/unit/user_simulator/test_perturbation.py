"""Unit tests for AudioPerturbator."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from eva.models.config import PerturbationConfig
from eva.user_simulator.perturbation import AudioPerturbator


def _make_pcm(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    n = int(duration_s * sample_rate)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate) * 16000).astype(np.int16)
    return samples.tobytes()


def _write_tmp_wav(tmp_path: Path, noise_type: str, duration_s: float = 30.0) -> Path:
    n = int(duration_s * 16000)
    samples = (np.random.randn(n) * 1000).astype(np.int16)
    wav_path = tmp_path / f"{noise_type}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return wav_path


@pytest.fixture()
def mock_assets(tmp_path, monkeypatch):
    _write_tmp_wav(tmp_path, "coffee_shop")
    _write_tmp_wav(tmp_path, "road_noise")
    monkeypatch.setattr("eva.user_simulator.perturbation._ASSETS_DIR", tmp_path)
    return tmp_path


class TestConnectionDegradation:
    def test_output_same_length(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert len(result) == len(pcm)

    def test_output_differs_from_input(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert result != pcm

    def test_codec_artifacts_changes_signal(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p._codec_artifacts(pcm)
        assert len(result) == len(pcm)

    def test_packet_loss_preserves_length(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p._apply_packet_loss(pcm)
        assert len(result) == len(pcm)

    def test_volume_fluctuation_preserves_length(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p._apply_volume_fluctuation(pcm)
        assert len(result) == len(pcm)


class TestBadConnectionStatic:
    def test_output_same_length(self):
        config = PerturbationConfig(background_noise="bad_connection_static")
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert len(result) == len(pcm)

    def test_output_differs_from_input(self):
        config = PerturbationConfig(background_noise="bad_connection_static")
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert result != pcm


class TestCoffeeShopNoise:
    def test_output_same_length(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert len(result) == len(pcm)

    def test_output_differs_from_input(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert result != pcm

    def test_noise_file_loops(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        pcm = _make_pcm(duration_s=40.0)
        result = p.apply(pcm)
        assert len(result) == len(pcm)


class TestRoadNoise:
    def test_output_same_length(self, mock_assets):
        config = PerturbationConfig(background_noise="road_noise", snr_db=10.0)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert len(result) == len(pcm)


class TestCombinedNoiseAndDegradation:
    def test_output_same_length(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0, connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert len(result) == len(pcm)

    def test_output_differs_from_input(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0, connection_degradation=True)
        p = AudioPerturbator(config)
        pcm = _make_pcm()
        result = p.apply(pcm)
        assert result != pcm


class TestMissingAsset:
    def test_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("eva.user_simulator.perturbation._ASSETS_DIR", tmp_path)
        config = PerturbationConfig(background_noise="coffee_shop")
        with pytest.raises(FileNotFoundError, match="download_noise_assets"):
            AudioPerturbator(config)


class TestEmptyInput:
    def test_empty_bytes_passthrough(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        assert p.apply(b"") == b""


class TestHasAmbientNoise:
    def test_true_when_background_noise_set(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        assert p.has_ambient_noise is True

    def test_false_when_only_connection_degradation(self):
        config = PerturbationConfig(connection_degradation=True)
        p = AudioPerturbator(config)
        assert p.has_ambient_noise is False


class TestGetAmbientChunk:
    def test_bad_connection_static_correct_length(self):
        config = PerturbationConfig(background_noise="bad_connection_static")
        p = AudioPerturbator(config)
        result = p.get_ambient_chunk(640)
        assert len(result) == 640

    def test_bad_connection_static_nonzero(self):
        config = PerturbationConfig(background_noise="bad_connection_static")
        p = AudioPerturbator(config)
        result = p.get_ambient_chunk(640)
        assert result != b"\x00" * 640

    def test_coffee_shop_correct_length(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        result = p.get_ambient_chunk(640)
        assert len(result) == 640

    def test_coffee_shop_nonzero(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        result = p.get_ambient_chunk(640)
        assert result != b"\x00" * 640

    def test_road_noise_correct_length(self, mock_assets):
        config = PerturbationConfig(background_noise="road_noise", snr_db=10.0)
        p = AudioPerturbator(config)
        result = p.get_ambient_chunk(640)
        assert len(result) == 640

    def test_cursor_advances_across_calls(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        chunk1 = p.get_ambient_chunk(640)
        chunk2 = p.get_ambient_chunk(640)
        assert chunk1 != chunk2

    def test_cursor_shared_with_apply(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        cursor_before = p._noise_cursor
        p.get_ambient_chunk(640)
        assert p._noise_cursor != cursor_before

    def test_noise_file_loops(self, mock_assets):
        config = PerturbationConfig(background_noise="coffee_shop", snr_db=15.0)
        p = AudioPerturbator(config)
        large_chunk = p.get_ambient_chunk(30 * 16000 * 2 + 640)
        assert len(large_chunk) == 30 * 16000 * 2 + 640
