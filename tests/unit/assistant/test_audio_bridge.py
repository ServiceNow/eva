"""Tests for shared audio bridge utilities.

Covers: PCM↔mulaw round-trip fidelity, PCM16 mixing with clipping,
and Twilio WebSocket protocol message round-trips.
"""

import audioop
import json
import math
import struct

import pytest

from eva.assistant.audio_bridge import (
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    pcm16_mix,
)


def _generate_mulaw_tone(freq_hz: int = 440, duration_ms: int = 100) -> bytes:
    sample_rate = 8000
    n_samples = sample_rate * duration_ms // 1000
    pcm_samples = [int(16000 * math.sin(2 * math.pi * freq_hz * i / sample_rate)) for i in range(n_samples)]
    pcm_bytes = struct.pack(f"<{n_samples}h", *pcm_samples)
    return audioop.lin2ulaw(pcm_bytes, 2)


def _rms(pcm_bytes: bytes) -> float:
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes)
    return math.sqrt(sum(s * s for s in samples) / n)


class TestAudioConversionRoundTrip:
    def test_mulaw_8k_pcm16_24k_round_trip(self):
        """Mulaw 8k -> pcm16 24k -> mulaw 8k preserves signal energy."""
        original = _generate_mulaw_tone(440, 100)

        pcm_24k = mulaw_8k_to_pcm16_24k(original)
        recovered = pcm16_24k_to_mulaw_8k(pcm_24k)

        assert len(recovered) == len(original)

        orig_pcm = audioop.ulaw2lin(original, 2)
        recov_pcm = audioop.ulaw2lin(recovered, 2)
        orig_rms = _rms(orig_pcm)
        recov_rms = _rms(recov_pcm)
        assert orig_rms > 0
        assert recov_rms / orig_rms == pytest.approx(1.0, abs=0.15)


class TestPcm16Mix:
    def test_adds_samples_and_clips_at_int16_boundaries(self):
        """Sample-wise addition with clipping; shorter track is zero-padded."""
        track_a = struct.pack("<2h", 30000, -30000)
        track_b = struct.pack("<2h", 10000, -10000)

        mixed = pcm16_mix(track_a, track_b)
        result = struct.unpack("<2h", mixed)
        assert result == (32767, -32768)

        short_track = struct.pack("<1h", 5000)
        long_track = struct.pack("<2h", 100, 200)
        mixed = pcm16_mix(short_track, long_track)
        result = struct.unpack("<2h", mixed)
        assert result == (5100, 200)


class TestTwilioProtocol:
    def test_create_and_parse_round_trip(self):
        """create_twilio_media_message -> parse_twilio_media_message recovers bytes."""
        audio = b"\x80\x90\xa0\xb0\xc0"
        msg = create_twilio_media_message("stream-1", audio)
        recovered = parse_twilio_media_message(msg)
        assert recovered == audio

        parsed = json.loads(msg)
        assert parsed["streamSid"] == "stream-1"

        assert parse_twilio_media_message(json.dumps({"event": "start"})) is None
        assert parse_twilio_media_message("not json at all {{{") is None
