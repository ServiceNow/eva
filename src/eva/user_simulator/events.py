"""User simulator event artifact compatibility helpers."""

from __future__ import annotations

from pathlib import Path

USER_SIMULATOR_EVENTS_FILENAME = "user_simulator_events.jsonl"
LEGACY_ELEVENLABS_EVENTS_FILENAME = "elevenlabs_events.jsonl"

_ROLE_TO_LEGACY = {
    "simulated_user": "elevenlabs_user",
    "assistant": "framework_agent",
}


def resolve_user_simulator_events_path(output_dir: Path, stored_path: str | None = None) -> Path | None:
    """Resolve the neutral event file, falling back to the legacy ElevenLabs artifact."""
    candidates: list[Path] = [
        output_dir / USER_SIMULATOR_EVENTS_FILENAME,
        output_dir / LEGACY_ELEVENLABS_EVENTS_FILENAME,
    ]
    if stored_path:
        stored = Path(stored_path)
        candidates.insert(0, output_dir / stored.name)
        candidates.append(stored)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_event_for_processor(event: dict) -> dict:
    """Normalize provider-neutral roles to the legacy processor vocabulary."""
    normalized = dict(event)
    role = normalized.get("user")
    if role in _ROLE_TO_LEGACY:
        normalized["user"] = _ROLE_TO_LEGACY[role]
    return normalized
