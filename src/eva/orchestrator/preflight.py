"""Pre-flight model checks — exercise each configured provider before a run starts.

A run can otherwise burn through every record and attempt before discovering that, say,
an LLM API key is expired or an STT endpoint is unreachable. ``run_preflight`` catches
those credential/connectivity failures up front by sending the cheapest possible real
request to each model the run will use.

Design notes:
- Probes reuse the same factories/clients the real run uses (``LiteLLMClient``,
  ``create_stt_service``/``create_tts_service``, ``create_audio_llm_client``), so a pass
  means the run's own construction path works.
- Probes are conservative: a component is reported as failed only on a definitive error
  (an exception or an ``ErrorFrame``). Ambiguity — no output, an unexpected-but-benign
  frame — is treated as a pass, so preflight never blocks an otherwise-valid run.
- Native-S2S backends (OpenAI Realtime, Grok Voice) are validated cheaply by
  *constructing* them via the ``BackendFactory`` (``_preflight_backends``): the backend's
  own ``__init__`` checks that required fields/keys are present, with no API call. This
  is the home for that validation (config-side validation was removed). Live S2S probing
  (an actual session) is still not supported.
"""

import asyncio
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from eva.assistant.pipeline.services import (
    create_audio_llm_client,
    create_stt_service,
    create_tts_service,
)
from eva.assistant.services.llm import LiteLLMClient
from eva.backend.factory import BackendFactory
from eva.models.config import PipelineType, RunConfig, get_model_alias_from_params
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Stateless; shared across preflight calls (see orchestrator.worker for the same pattern).
_BACKEND_FACTORY = BackendFactory()

# 0.3s of 16 kHz mono 16-bit silence — enough to make a streaming STT service open its
# connection (and reveal an auth failure) without depending on actual speech content.
_SILENCE_SAMPLE_RATE = 16000
_SILENCE_PCM = b"\x00\x00" * (_SILENCE_SAMPLE_RATE // 3)


class PreflightError(RuntimeError):
    """Raised when a preflight probe fails, to abort before any simulations run."""


@dataclass
class ProbeResult:
    """Outcome of probing a single model/provider."""

    model_type: str  # "LLM", "STT", "TTS", "AUDIO_LLM"
    model_alias: str
    ok: bool
    detail: str = ""


class _FrameCapture(FrameProcessor):
    """Pass-through processor that records every frame it sees."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def _drive_pipecat_service(service: FrameProcessor, input_frames: list[Frame]) -> tuple[bool, str]:
    """Run a pipecat STT/TTS service through a minimal pipeline and report success.

    The service is wrapped between two capture processors: errors travel upstream and
    output (audio/transcripts) travels downstream, so both are observed. The pipeline
    auto-sends a ``StartFrame`` (which is when streaming services connect and surface
    auth failures as an ``ErrorFrame``); the trailing ``EndFrame`` shuts it down cleanly.

    Returns ``(ok, detail)`` — ``ok=False`` only when an ``ErrorFrame`` is seen.
    """
    head, tail = _FrameCapture(), _FrameCapture()
    task = PipelineWorker(Pipeline([head, service, tail]))
    runner = WorkerRunner(handle_sigint=False, force_gc=True)

    async def feed() -> None:
        await task.queue_frames([*input_frames, EndFrame()])

    try:
        await runner.add_workers(task)
        await asyncio.gather(runner.run(), feed())
    finally:
        if not task.has_finished():
            try:
                await task.cancel()
            except Exception:
                pass

    errors = [f for f in head.frames + tail.frames if isinstance(f, ErrorFrame)]
    if errors:
        return False, str(errors[0].error)
    return True, ""


async def _probe_llm(config: RunConfig) -> tuple[bool, str]:
    client = LiteLLMClient(model=config.model.llm)
    await client.complete([{"role": "user", "content": "ping"}], max_retries=0)
    return True, ""


async def _probe_audio_llm(config: RunConfig) -> tuple[bool, str]:
    client = create_audio_llm_client(
        config.model.audio_llm, config.model.audio_llm_params, language=config.language.value
    )
    # Send a real (silent) audio turn so the probe exercises the actual ALM path.
    message = client.build_audio_user_message(_SILENCE_PCM, source_sample_rate=_SILENCE_SAMPLE_RATE)
    await client.complete([message])
    return True, ""


async def _probe_stt(config: RunConfig) -> tuple[bool, str]:
    service = create_stt_service(config.model.stt, config.model.stt_params, language_code=config.language.value)
    if service is None:
        return True, ""
    audio = InputAudioRawFrame(audio=_SILENCE_PCM, sample_rate=_SILENCE_SAMPLE_RATE, num_channels=1)
    return await _drive_pipecat_service(service, [audio])


async def _probe_tts(config: RunConfig) -> tuple[bool, str]:
    service = create_tts_service(config.model.tts, config.model.tts_params, language_code=config.language.value)
    if service is None:
        return True, ""
    return await _drive_pipecat_service(service, [TTSSpeakFrame("ok")])


async def _guard(
    model_type: str, model_alias: str, probe: Coroutine[Any, Any, tuple[bool, str]], timeout: float
) -> ProbeResult:
    """Run one probe under a timeout, converting any failure into a ``ProbeResult``."""
    try:
        ok, detail = await asyncio.wait_for(probe, timeout)
        return ProbeResult(model_type, model_alias, ok, detail)
    except TimeoutError:
        return ProbeResult(model_type, model_alias, False, f"timed out after {timeout:.0f}s")
    except Exception as e:
        msg = str(e).strip() or type(e).__name__
        return ProbeResult(model_type, model_alias, False, msg[:200])


async def _run_preflight(config: RunConfig) -> list[ProbeResult]:
    """Probe every model the run will use; return one ``ProbeResult`` per component.

    S2S runs are skipped (no live probe yet) — only a log line is emitted.
    """
    timeout = config.preflight_timeout_seconds
    model = config.model
    probes: list[tuple[str, str, Awaitable[tuple[bool, str]]]] = []

    match model.pipeline_type:
        case PipelineType.CASCADE:
            probes.append(("LLM", model.llm, _probe_llm(config)))
            probes.append(("STT", get_model_alias_from_params(model.stt_params), _probe_stt(config)))
            probes.append(("TTS", get_model_alias_from_params(model.tts_params), _probe_tts(config)))
        case PipelineType.AUDIO_LLM:
            probes.append(("AUDIO_LLM", get_model_alias_from_params(model.audio_llm_params), _probe_audio_llm(config)))
            probes.append(("TTS", get_model_alias_from_params(model.tts_params), _probe_tts(config)))
        case PipelineType.S2S:
            # Backend construction is validated separately (_preflight_backends); a live
            # S2S session probe is not yet supported.
            logger.info("Pre-flight: S2S live probe not yet supported (construction validated)")
            return []

    logger.info(f"Pre-flight: checking {len(probes)} model(s) before the run starts...")
    return await asyncio.gather(*(_guard(model_type, alias, probe, timeout) for model_type, alias, probe in probes))


def _check_backend_construction(label: str, name: str, backend_args: dict[str, Any]) -> str | None:
    """Validate a native-S2S provider by constructing its backend (no network).

    The backend's own ``__init__`` checks that required fields/keys resolve, so a
    successful construction means the run's construction path works. Returns an error
    string for a *misconfigured factory backend* (missing key, absent model), or
    ``None`` when it's fine — including when the factory doesn't back this provider
    (``create()`` -> ``None``): that's a legacy/non-factory provider (e.g. ElevenLabs
    Conversational AI, or a cascade pipeline) validated elsewhere, not here.
    """
    try:
        _BACKEND_FACTORY.create(name, backend_args)
    except Exception as e:
        return f"{label} {name!r}: {str(e).strip()[:200] or type(e).__name__}"
    return None


def _preflight_backends(config: RunConfig) -> None:
    """Cheap, network-free validation that the configured S2S assistant backend constructs.

    Only the S2S assistant framework runs through the ``BackendFactory``; constructing it
    validates required fields/keys with no API call. Cascade/audio-LLM assistants (pipecat
    services) are left to the live probes. The user simulator is on its legacy stack and
    validated by config (not a backend), so it is not checked here.
    """
    if config.model.pipeline_type != PipelineType.S2S:
        return
    assistant_args = {
        **(config.model.s2s_params or {}),
        "parallel_tool_calls": config.model.parallel_tool_calls,
    }
    if err := _check_backend_construction("assistant framework", config.framework, assistant_args):
        raise PreflightError(f"backend configuration invalid:\n{err}")


async def run_preflight(config: RunConfig) -> None:
    """Validate backends, then probe models; raise ``PreflightError`` on any failure.

    The cheap, network-free backend-construction check (``_preflight_backends``) ALWAYS
    runs -- ``--no-preflight`` only skips the live model probes (the ones that make real
    API calls), never the config validation. Call this immediately before launching
    simulations.
    """
    _preflight_backends(config)  # always: cheap, no network
    if not config.preflight:
        return
    results = await _run_preflight(config)
    failed = [r for r in results if not r.ok]
    if failed:
        detail = "\n".join(f"{r.model_type} ({r.model_alias}): {r.detail}" for r in failed)
        raise PreflightError(f"preflight check failed:\n{detail}")
