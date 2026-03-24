"""Suppress known harmless warnings before noisy third-party imports."""

import warnings

# Suppress pyannote's torchcodec/FFmpeg warning at module-import time.
# pyannote.audio.core.io unconditionally tries to load torchcodec when it is
# first imported (triggered by `import whisperx`). The FFmpeg shared libraries
# are not present in this environment, so the attempt always fails and emits a
# UserWarning. This is harmless because we pass all audio as preloaded in-memory
# dicts {'waveform': Tensor, 'sample_rate': int}, bypassing torchcodec entirely.
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
