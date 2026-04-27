"""Shared audio I/O helpers."""

import wave
from pathlib import Path

from eva.utils.logging import get_logger

logger = get_logger(__name__)


def save_pcm_as_wav(
    audio_data: bytes,
    file_path: Path,
    sample_rate: int,
    num_channels: int,
    sample_width: int = 2,
) -> None:
    """Save raw PCM audio data to a WAV file."""
    try:
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        logger.debug(f"Audio saved to {file_path} ({len(audio_data)} bytes)")
    except Exception as e:
        logger.error(f"Error saving audio to {file_path}: {e}")
