"""Schema constants for the EVA config editor.

Variable metadata (widget types, options, ranges, tooltips, conditions) is
now encoded directly in .env.example using annotation prefixes (#i, #d, #e,
#r, #g, #x, #v).  This module retains only things that are inherently
editor-behaviour rather than file-structure:

- Tab group name constants and ordering.
- Mutex radio-button definitions (pipeline mode, perturbation mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field

GROUP_API_CONFIGS = "API Configs"
GROUP_VOICE_PIPELINE = "Voice Pipeline"
GROUP_DEPLOYMENTS = "LiteLLM Deployments"
GROUP_RUNTIME = "Framework & Runtime"
GROUP_TURN = "Turn Detection & VAD"
GROUP_PERTURBATIONS = "User Config"
GROUP_DEBUG = "Debug & Logging"
GROUP_MISC = "Misc / Unmapped"

GROUPS: list[str] = [
    GROUP_API_CONFIGS,
    GROUP_VOICE_PIPELINE,
    GROUP_DEPLOYMENTS,
    GROUP_RUNTIME,
    GROUP_TURN,
    GROUP_PERTURBATIONS,
    GROUP_DEBUG,
]


@dataclass
class MutexRadio:
    """A UI radio button that enforces mutual exclusion among a set of vars."""

    state_key: str  # st.session_state key managed by this radio
    group: str  # which tab renders this radio
    label: str
    options: list[str]
    help: str = ""
    default: str = field(default="")

    def __post_init__(self) -> None:
        if not self.default and self.options:
            self.default = self.options[0]


MUTEX_RADIOS: list[MutexRadio] = [
    MutexRadio(
        state_key="pipeline_mode",
        group=GROUP_VOICE_PIPELINE,
        label="Pipeline mode",
        options=["LLM", "S2S", "AudioLLM"],
        help="LLM = STT+LLM+TTS. S2S = speech-to-speech model. AudioLLM = audio-input LLM + TTS.",
        default="LLM",
    ),
    MutexRadio(
        state_key="perturbation_mode",
        group=GROUP_PERTURBATIONS,
        label="User mode",
        options=["None", "Language", "Accent", "Behavior"],
        help="Language, Accent, and Behavior are mutually exclusive — each claims the ElevenLabs agent ID slot.",
        default="None",
    ),
]
