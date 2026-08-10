"""Generate trimmed (silence-collapsed) audio for the speech-fidelity labeling app.

The app (apps/metrics_labeler.py) looks for "{audio_base}_trimmed.wav" next to each
record's full audio in the audio dir, purely as a listening convenience (skip dead
air while labeling). It's optional — the app just shows a warning if missing — but
makes listening much faster on longer recordings.

Reuses the ACTUAL tts_fidelity judge's trimming method (SpeechFidelityBaseMetric.
_trim_silence) rather than reimplementing the logic, so labelers hear exactly what
the judge heard — same silence thresholds, same padding, same behavior.

Usage:
    python scripts/generate_trimmed_audio.py --audio-dir agent_speech_fidelity_audios
    python scripts/generate_trimmed_audio.py --audio-dir agent_speech_fidelity_audios --force
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from eva.metrics.registry import get_global_registry
from eva.metrics.utils import load_audio_file
from eva.utils.logging import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trimmed audio for the speech-fidelity labeling app.")
    parser.add_argument("--audio-dir", required=True, help="Dir containing {id}_expected_rating_{r}.wav files")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a trimmed file already exists")
    args = parser.parse_args()

    registry = get_global_registry()
    metric_cls = registry.get("tts_fidelity")
    if metric_cls is None:
        sys.exit("Error: metric 'tts_fidelity' not found in registry.")
    metric = metric_cls()

    audio_dir = Path(args.audio_dir)
    full_files = sorted(
        p for p in audio_dir.glob("*.wav") if not p.stem.endswith("_trimmed")
    )
    if not full_files:
        sys.exit(f"Error: no .wav files found in {audio_dir}")

    n_done, n_skipped, n_failed = 0, 0, 0
    for src in full_files:
        trimmed_path = audio_dir / f"{src.stem}_trimmed.wav"
        if trimmed_path.exists() and not args.force:
            n_skipped += 1
            continue

        segment = load_audio_file(src)
        if segment is None:
            logger.error(f"{src.name}: failed to load audio")
            n_failed += 1
            continue

        dummy_context = SimpleNamespace(record_id=src.stem, output_dir=None)
        trimmed = metric._trim_silence(segment, dummy_context)
        trimmed.export(str(trimmed_path), format="wav")
        n_done += 1

    logger.info(f"Generated {n_done} trimmed file(s), skipped {n_skipped} (already existed), failed {n_failed}.")


if __name__ == "__main__":
    main()
