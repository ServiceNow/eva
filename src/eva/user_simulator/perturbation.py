"""Audio perturbations applied to simulated user speech.

Perturbations are injected after ElevenLabs generates 16 kHz PCM16 audio and
before it is sent to the assistant, making them framework-agnostic.

Two audio-level perturbation modes are supported:
- background_noise: mix in a noise file (or synthetic static) at a target SNR,
  applied to both speech and silence periods
- connection_degradation: compound VoIP degradation (codec quantisation, packet
  loss, volume fluctuation) applied on top of any background noise

Accent and behavior perturbations are handled in client.py (agent ID / persona).
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from eva.models.config import PerturbationConfig

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[no-reuse-def]

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes per sample

_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "noise"

_FILE_NOISE_TYPES: set[str] = {
    "coffee_shop",
    "airport_gate",
    "road_noise",
    "nyc_street",
    "background_music",
    "loud_construction",
    "baby_crying",
}

_BAD_CONNECTION_PACKET_LOSS_PROB = 0.03
_BAD_CONNECTION_SEGMENT_BYTES = 640  # 20 ms at 16 kHz 16-bit PCM
_BAD_CONNECTION_GAIN_LOW = 0.5
_BAD_CONNECTION_GAIN_HIGH = 1.2

_REFERENCE_SPEECH_RMS: float = 3000.0  # typical 16-bit conversational speech (~-21 dBFS)


def _load_noise_wav(noise_type: str) -> np.ndarray:
    path = _ASSETS_DIR / f"{noise_type}.wav"
    if not path.exists():
        raise FileNotFoundError(
            f"Noise asset not found: {path}\n"
            f"Run 'python scripts/download_noise_assets.py' to prepare the required files."
        )
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Noise file must be mono: {path}")
        if wf.getsampwidth() != _SAMPLE_WIDTH:
            raise ValueError(f"Noise file must be 16-bit PCM: {path}")
        if wf.getframerate() != _SAMPLE_RATE:
            raise ValueError(f"Noise file must be {_SAMPLE_RATE} Hz: {path}")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    rms_speech = np.sqrt(np.mean(speech**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    if rms_noise < 1e-9:
        return speech
    target_rms_noise = rms_speech * (10 ** (-snr_db / 20.0))
    noise_scaled = noise * (target_rms_noise / rms_noise)
    mixed = speech + noise_scaled
    return np.clip(mixed, -32768, 32767)


def _pull_noise_chunk(noise: np.ndarray, cursor: int, n: int) -> tuple[np.ndarray, int]:
    """Pull n samples from a looping noise array starting at cursor.

    Returns (chunk, new_cursor).
    """
    if cursor + n > len(noise):
        chunk = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            remaining = len(noise) - cursor
            take = min(remaining, n - filled)
            chunk[filled : filled + take] = noise[cursor : cursor + take]
            filled += take
            cursor = (cursor + take) % len(noise)
    else:
        chunk = noise[cursor : cursor + n].copy()
        cursor += n
    return chunk, cursor


class AudioPerturbator:
    """Applies audio-level perturbations to 16 kHz 16-bit PCM mono bytes.

    Handles background_noise mixing (file-based or synthetic static) and
    optional connection_degradation (codec, packet loss, volume fluctuation).
    Accent and behavior perturbations are handled separately in client.py.
    """

    def __init__(self, config: PerturbationConfig) -> None:
        self._config = config
        self._noise: np.ndarray | None = None
        self._noise_cursor: int = 0

        if config.background_noise is not None and config.background_noise in _FILE_NOISE_TYPES:
            self._noise = _load_noise_wav(str(config.background_noise))

    @property
    def has_ambient_noise(self) -> bool:
        """True when silence periods should be replaced with ambient noise."""
        return self._config.background_noise is not None

    def apply(self, pcm_bytes: bytes) -> bytes:
        """Apply all configured perturbations to a PCM chunk."""
        if not pcm_bytes:
            return pcm_bytes

        result = pcm_bytes

        if self._config.background_noise is not None:
            if self._config.background_noise == "bad_connection_static":
                result = self._apply_static(result)
            else:
                result = self._apply_file_noise(result)

        if self._config.connection_degradation:
            result = self._apply_connection_degradation(result)

        return result

    def get_ambient_chunk(self, n_bytes: int) -> bytes:
        """Return n_bytes of ambient noise PCM to replace silence frames.

        The noise cursor is shared with apply(), so speech and silence periods
        consume the noise file as one continuous stream.

        Args:
            n_bytes: Number of PCM bytes to generate (must be even).

        Returns:
            PCM bytes at 16 kHz 16-bit mono.
        """
        n_samples = n_bytes // _SAMPLE_WIDTH
        target_rms = _REFERENCE_SPEECH_RMS * (10 ** (-self._config.snr_db / 20.0))

        if self._config.background_noise == "bad_connection_static":
            samples = np.random.normal(0.0, target_rms, size=n_samples)
            return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

        assert self._noise is not None
        noise_chunk, self._noise_cursor = _pull_noise_chunk(self._noise, self._noise_cursor, n_samples)

        rms_noise = np.sqrt(np.mean(noise_chunk**2))
        if rms_noise < 1e-9:
            return b"\x00" * n_bytes

        scaled = noise_chunk * (target_rms / rms_noise)
        return np.clip(scaled, -32768, 32767).astype(np.int16).tobytes()

    def _apply_file_noise(self, pcm_bytes: bytes) -> bytes:
        assert self._noise is not None
        speech = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        noise_chunk, self._noise_cursor = _pull_noise_chunk(self._noise, self._noise_cursor, len(speech))
        mixed = _mix_at_snr(speech, noise_chunk, self._config.snr_db)
        return mixed.astype(np.int16).tobytes()

    def _apply_static(self, pcm_bytes: bytes) -> bytes:
        speech = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(speech**2))
        if rms < 1e-9:
            rms = _REFERENCE_SPEECH_RMS
        sigma = rms * (10 ** (-self._config.snr_db / 20.0))
        noise = np.random.normal(0.0, sigma, size=len(speech))
        mixed = np.clip(speech + noise, -32768, 32767)
        return mixed.astype(np.int16).tobytes()

    def _apply_connection_degradation(self, pcm_bytes: bytes) -> bytes:
        pcm = self._codec_artifacts(pcm_bytes)
        pcm = self._apply_packet_loss(pcm)
        pcm = self._apply_volume_fluctuation(pcm)
        return pcm

    def _codec_artifacts(self, pcm_bytes: bytes) -> bytes:
        mulaw = audioop.lin2ulaw(pcm_bytes, _SAMPLE_WIDTH)
        return audioop.ulaw2lin(mulaw, _SAMPLE_WIDTH)

    def _apply_packet_loss(self, pcm_bytes: bytes) -> bytes:
        data = bytearray(pcm_bytes)
        seg = _BAD_CONNECTION_SEGMENT_BYTES
        for offset in range(0, len(data), seg):
            if np.random.random() < _BAD_CONNECTION_PACKET_LOSS_PROB:
                end = min(offset + seg, len(data))
                data[offset:end] = b"\x00" * (end - offset)
        return bytes(data)

    def _apply_volume_fluctuation(self, pcm_bytes: bytes) -> bytes:
        gain = np.random.uniform(_BAD_CONNECTION_GAIN_LOW, _BAD_CONNECTION_GAIN_HIGH)
        speech = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        mixed = np.clip(speech * gain, -32768, 32767)
        return mixed.astype(np.int16).tobytes()
