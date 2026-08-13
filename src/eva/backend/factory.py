"""Factory that constructs ``Backend`` instances by provider name.

Mirrors the shape of ``eva.user_simulator.factory.create_user_simulator`` (lazy
per-provider imports keyed off a provider name) but is
provider-and-role-agnostic: the same factory builds a backend for either an
``AssistantRole`` or a ``UserRole``, since a ``Backend`` has no role knowledge
(that's the whole point of the split -- see docs/refactor-step1.md, "lets any
backend act as either role").

A single concrete class -- there is no abstract base, since there is only ever
one factory. ``create`` returns ``None`` for a provider that has not been
migrated onto the ``Backend`` contract; the worker uses that as the signal to
fall back to the legacy server/simulator for that provider.
"""

from __future__ import annotations

from typing import Any

from eva.backend.base import Backend


class BackendFactory:
    """Constructs a ``Backend`` for a named provider from a config blob.

    Providers are added as their backends are migrated onto the ``Backend``
    contract: add a lazy-import branch in ``create``. Each provider's SDK is
    imported only when that provider is selected, so unused providers need not
    be importable. ``create`` returning ``None`` is the single signal for "not a
    native backend" -- there is no parallel list of supported names to maintain.
    """

    def create(self, name: str, config: dict[str, Any]) -> Backend | None:
        """Construct a not-yet-opened ``Backend``, or ``None`` if not yet migrated.

        Args:
            name: Provider identifier (e.g. ``"openai_realtime"``,
                ``"gemini_live"``, ``"elevenlabs"``, ``"cascade"``).
            config: Provider-specific configuration understood by that
                backend. This factory does not validate the shape of
                ``config`` beyond dispatching on ``name`` -- each ``Backend``
                subclass validates its own config and assembles its own
                provider session (the caller hand-builds no provider JSON).

        Returns:
            A constructed ``Backend`` for a migrated provider, not yet
            ``open()``ed (construction and session establishment are separate
            steps, so a ``Role`` can build its backend early and open the
            session later). ``None`` if ``name`` is not a migrated provider --
            the assistant path treats that as an error (unported = unusable);
            the (unported) legacy server survives only as reference.
        """
        if name == "openai_realtime":
            from eva.backend.openai_realtime import OpenAIRealtimeBackend

            return OpenAIRealtimeBackend(config=config)
        if name == "grok_voice":
            from eva.backend.grok_voice import GrokVoiceBackend

            return GrokVoiceBackend(config=config)
        if name == "elevenlabs":
            from eva.backend.elevenlabs import ElevenLabsBackend

            return ElevenLabsBackend(config=config)
        if name == "gemini_live":
            from eva.backend.gemini_live import GeminiLiveBackend

            return GeminiLiveBackend(config=config)
        return None
