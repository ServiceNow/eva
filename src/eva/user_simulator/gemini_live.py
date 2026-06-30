"""Gemini Live implementation of the EVA simulated caller.

Mirrors the OpenAI Realtime caller but drives a second Gemini Live session as the
simulated user. Audio flows:

    Assistant (8 kHz mulaw, via the audio bridge)
        -> 16 kHz PCM16 -> Gemini Live realtime input
    Gemini Live output (24 kHz PCM16)
        -> 16 kHz PCM16 -> audio bridge -> assistant

Gemini's automatic activity detection handles turn-taking, so unlike the OpenAI
caller there is no manual response sequencing. An ``end_call`` function tool lets
the caller hang up under the same conditions as the OpenAI caller.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.models.config import GeminiLiveSimulatorConfig, PerturbationConfig
from eva.user_simulator.audio_bridge import BotToBotAudioBridge
from eva.user_simulator.base import END_CALL_DESCRIPTION, AbstractUserSimulator
from eva.utils.audio_utils import save_pcm_as_wav
from eva.utils.logging import get_logger

logger = get_logger(__name__)

GEMINI_SAMPLE_RATE = 24000
BRIDGE_SAMPLE_RATE = 16000
ASSISTANT_SAMPLE_RATE = 8000
_PERSONA_GENDER = {1: "F", 2: "M"}

# Known Gemini Live prebuilt voice names. An unrecognised voice is accepted
# silently by the API and falls back to a default voice, so warn up front.
# Half-cascade models (e.g. gemini-3.1-flash-live-preview) support the first
# eight; native-audio models support the full set.
_KNOWN_GEMINI_VOICES = frozenset(
    {
        "Aoede",
        "Charon",
        "Fenrir",
        "Kore",
        "Leda",
        "Orus",
        "Puck",
        "Zephyr",
        "Achernar",
        "Achird",
        "Algenib",
        "Algieba",
        "Alnilam",
        "Autonoe",
        "Callirrhoe",
        "Despina",
        "Enceladus",
        "Erinome",
        "Gacrux",
        "Iapetus",
        "Laomedeia",
        "Pulcherrima",
        "Rasalgethi",
        "Sadachbia",
        "Sadaltager",
        "Schedar",
        "Sulafat",
        "Umbriel",
        "Vindemiatrix",
        "Zubenelgenubi",
    }
)


class GeminiLiveUserSimulator(AbstractUserSimulator):
    """Use a Gemini Live session as EVA's simulated caller."""

    def __init__(
        self,
        current_date_time: str,
        persona_config: dict,
        goal: dict,
        server_url: str,
        output_dir: Path,
        agent_id: str,
        timeout: int = 600,
        perturbation_config: PerturbationConfig | None = None,
        language: str = "en",
        *,
        simulator_config: GeminiLiveSimulatorConfig,
    ) -> None:
        super().__init__(
            current_date_time=current_date_time,
            persona_config=persona_config,
            goal=goal,
            server_url=server_url,
            output_dir=output_dir,
            agent_id=agent_id,
            timeout=timeout,
            perturbation_config=perturbation_config,
            language=language,
            provider="gemini_live",
        )
        if perturbation_config and perturbation_config.accent is not None:
            raise ValueError("Gemini Live caller does not support ElevenLabs-specific accent variants")
        self.simulator_config = simulator_config
        self._assistant_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._caller_transcript_parts: list[str] = []
        self._caller_audio_seen = False
        self._input_resampler_state = None
        self._output_resampler_state = None

    @property
    def caller_model(self) -> str:
        return self.simulator_config.model

    @property
    def caller_voice(self) -> str:
        gender = _PERSONA_GENDER.get(self.persona_config.get("user_persona_id"))
        if gender == "M":
            return self.simulator_config.male_voice
        return self.simulator_config.female_voice

    def _build_live_config(self) -> types.LiveConnectConfig:
        voice = self.caller_voice
        if voice.lower() not in {v.lower() for v in _KNOWN_GEMINI_VOICES}:
            logger.warning(
                f"Configured Gemini caller voice {voice!r} is not a recognised prebuilt voice; "
                f"Gemini will silently fall back to a default voice. Known voices: "
                f"{', '.join(sorted(_KNOWN_GEMINI_VOICES))}"
            )
            self.event_logger.log_event("invalid_voice", {"voice": voice})
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=self._build_prompt(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="end_call",
                            description=END_CALL_DESCRIPTION,
                            parameters=types.Schema(type="OBJECT", properties={}),
                            behavior=types.Behavior.BLOCKING,
                        )
                    ]
                )
            ],
        )

    def _create_client(self, api_key: str | None) -> genai.Client:
        """Create a google-genai client.

        There are two distinct auth backends and you need exactly ONE of them —
        a service-account JSON and an API key are not combined:

        1. Vertex AI — GOOGLE_CLOUD_PROJECT (+ GOOGLE_CLOUD_LOCATION). Authenticated
           via Application Default Credentials, i.e. GOOGLE_APPLICATION_CREDENTIALS
           (service-account JSON), workload identity, or ``gcloud auth``. The API
           key is ignored on this path.
        2. Gemini Developer API — GEMINI_API_KEY / GOOGLE_API_KEY. The
           service-account JSON is ignored on this path.

        Vertex is preferred when a project is configured so service-account
        deployments work; otherwise we fall back to the API key, then to the
        SDK's own default resolution (e.g. GOOGLE_GENAI_USE_VERTEXAI=true + ADC).
        """
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            logger.info(f"Gemini caller using Vertex AI (project={project}, location={location})")
            return genai.Client(vertexai=True, project=project, location=location)
        if api_key:
            logger.info("Gemini caller using Developer API key")
            return genai.Client(api_key=api_key)
        logger.warning("No explicit Gemini credentials; relying on google-genai default resolution")
        return genai.Client()

    async def run_conversation(self) -> str:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if (
            not api_key
            and not os.environ.get("GOOGLE_CLOUD_PROJECT")
            and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        ):
            raise ValueError(
                "Gemini Live caller requires one of GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT, "
                "or GOOGLE_APPLICATION_CREDENTIALS"
            )

        try:
            await self._run_gemini_conversation(api_key)
        except Exception as exc:
            logger.error(f"Gemini caller simulation error: {exc}", exc_info=True)
            self._end_reason = "error"
            self.event_logger.log_error(str(exc))
            if self._audio_interface is not None:
                with suppress(Exception):
                    await self._audio_interface.stop_async()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})
        finally:
            self.event_logger.save()
        return self._end_reason

    async def _run_gemini_conversation(self, api_key: str | None) -> None:
        conversation_id = self.output_dir.name
        self._audio_interface = BotToBotAudioBridge(
            websocket_uri=self.server_url,
            conversation_id=conversation_id,
            record_callback=self._record_audio,
            event_logger=self.event_logger,
            conversation_done_callback=self._on_conversation_end,
            perturbator=self._perturbator,
            disconnect_reason="assistant_disconnect",
        )
        await self._audio_interface.start_async()
        self._audio_interface.start(self._on_assistant_audio)
        self.event_logger.log_connection_state(
            "connected",
            {
                "server_url": self.server_url,
                "caller_provider": self.provider,
                "caller_model": self.caller_model,
                "caller_voice": self.caller_voice,
                "caller_input_format": "audio/pcm;rate=16000",
                "caller_output_format": f"audio/pcm;rate={GEMINI_SAMPLE_RATE}",
                "assistant_input_transport": "audio/pcmu_8000hz",
                "caller_turn_detection": "gemini_automatic_activity_detection",
            },
        )

        client = self._create_client(api_key)
        forward_task: asyncio.Task | None = None
        listener_task: asyncio.Task | None = None
        completion_task: asyncio.Task | None = None
        try:
            async with client.aio.live.connect(model=self.caller_model, config=self._build_live_config()) as session:
                self.event_logger.log_connection_state("session_started")
                forward_task = asyncio.create_task(self._forward_assistant_audio(session))
                listener_task = asyncio.create_task(self._listen_for_caller_events(session))
                completion_task = asyncio.create_task(self._wait_for_conversation_end())

                await self._wait_for_session_completion(completion_task, forward_task, listener_task)

                # Allow final goodbye audio and transcripts to flush before closing.
                await asyncio.sleep(4.0)
        finally:
            for task in (completion_task, forward_task, listener_task):
                if task is not None:
                    await self._cancel_background_task(task)
            await self._audio_interface.stop_async()
            self._save_user_audio()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

    @staticmethod
    async def _cancel_background_task(task: asyncio.Task) -> None:
        """Cancel and consume a background task without interrupting cleanup."""
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    async def _wait_for_conversation_end(self) -> None:
        try:
            await asyncio.wait_for(self._conversation_done.wait(), timeout=self.timeout)
        except TimeoutError:
            self.event_logger.log_event("timeout", {"duration": self.timeout})
            self._on_conversation_end("timeout")

    async def _wait_for_session_completion(
        self,
        completion_task: asyncio.Task,
        forward_task: asyncio.Task,
        listener_task: asyncio.Task,
    ) -> None:
        done, _ = await asyncio.wait(
            {completion_task, forward_task, listener_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if completion_task in done:
            return

        finished_task = next(iter(done))
        if self._conversation_done.is_set():
            await completion_task
            return

        exception = finished_task.exception()
        if exception is not None:
            raise exception
        task_name = "listener" if finished_task is listener_task else "audio forwarder"
        raise RuntimeError(f"Gemini Live {task_name} stopped unexpectedly")

    def _on_assistant_audio(self, mulaw_audio: bytes) -> None:
        # Don't echo the caller's own audio back into Gemini while it is speaking.
        if mulaw_audio and not self._caller_audio_is_playing():
            self._assistant_audio_queue.put_nowait(mulaw_audio)

    def _caller_audio_is_playing(self) -> bool:
        if self._audio_interface is None:
            return False
        return self._audio_interface.is_caller_playing()

    async def _forward_assistant_audio(self, session: Any) -> None:
        while True:
            mulaw_audio = await self._assistant_audio_queue.get()
            if not mulaw_audio:
                continue
            pcm16_8k = audioop.ulaw2lin(mulaw_audio, 2)
            pcm16_16k, self._input_resampler_state = audioop.ratecv(
                pcm16_8k,
                2,
                1,
                ASSISTANT_SAMPLE_RATE,
                BRIDGE_SAMPLE_RATE,
                self._input_resampler_state,
            )
            with suppress(Exception):
                await session.send_realtime_input(audio=types.Blob(data=pcm16_16k, mime_type="audio/pcm;rate=16000"))

    async def _listen_for_caller_events(self, session: Any) -> None:
        try:
            while not self._conversation_done.is_set():
                try:
                    response = await asyncio.wait_for(session._receive(), timeout=2.0)
                except TimeoutError:
                    continue
                if response is None:
                    continue
                await self._handle_caller_event(session, response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Gemini caller event loop error: {exc}", exc_info=True)
            self.event_logger.log_error(str(exc))
            self._on_conversation_end("error")

    async def _handle_caller_event(self, session: Any, response: Any) -> None:
        sc = getattr(response, "server_content", None)
        if sc is not None:
            if sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        self._emit_caller_audio(bytes(part.inline_data.data))
            if sc.input_transcription and (sc.input_transcription.text or "").strip():
                # Input transcription = what Gemini heard, i.e. the assistant.
                self._on_assistant_speaks(sc.input_transcription.text.strip())
            if sc.output_transcription and (sc.output_transcription.text or "").strip():
                # Output transcription = the simulated caller's own speech.
                self._caller_transcript_parts.append(sc.output_transcription.text.strip())
            if sc.turn_complete:
                self._flush_caller_transcript()
                self._flush_caller_output()
                self._output_resampler_state = None

        tool_call = getattr(response, "tool_call", None)
        if tool_call:
            for fc in tool_call.function_calls:
                if fc.name == "end_call":
                    self.event_logger.log_event("tool_call", {"name": "end_call", "arguments": dict(fc.args or {})})
                    with suppress(Exception):
                        await session.send_tool_response(
                            function_responses=[
                                types.FunctionResponse(id=fc.id, name=fc.name, response={"status": "ended"})
                            ]
                        )
                    self._flush_caller_transcript()
                    self._on_conversation_end("goodbye")

    def _emit_caller_audio(self, pcm16_24k: bytes) -> None:
        if self._audio_interface is None or len(pcm16_24k) < 2:
            return
        pcm16_16k, self._output_resampler_state = audioop.ratecv(
            pcm16_24k,
            2,
            1,
            GEMINI_SAMPLE_RATE,
            BRIDGE_SAMPLE_RATE,
            self._output_resampler_state,
        )
        self._audio_interface.output(pcm16_16k)
        self._caller_audio_seen = True

    def _flush_caller_transcript(self) -> None:
        transcript = " ".join(part for part in self._caller_transcript_parts if part).strip()
        self._caller_transcript_parts.clear()
        if transcript:
            self._on_user_speaks(transcript)

    def _flush_caller_output(self) -> None:
        if self._caller_audio_seen and self._audio_interface is not None:
            self._audio_interface.output(b"\x00\x00")
            self._caller_audio_seen = False

    def _save_user_audio(self) -> None:
        if not self._user_clean_audio_chunks:
            return
        save_pcm_as_wav(
            b"".join(self._user_clean_audio_chunks),
            self.output_dir / "audio_user_clean.wav",
            sample_rate=BRIDGE_SAMPLE_RATE,
            num_channels=1,
        )
