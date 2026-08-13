"""Gemini Live ``Backend``: normalizing adapter for one Google Gemini Live session.

Wraps a single Gemini Live session (google-genai ``client.aio.live``) behind the
role-agnostic ``Backend`` contract (see ``eva.backend.base``), mirroring
``eva.backend.openai_realtime``. All provider-specific work lives here so roles
stay generic:

- session lifecycle: connect / stream audio in / events out / tool results / close;
- ``LiveConnectConfig`` assembly (voice, VAD, transcription, language) from the
  worker-supplied flat config -- roles never build provider config;
- tool-schema translation: generic ``{name, description, parameters}`` specs ->
  Gemini ``types.Tool`` / ``FunctionDeclaration``;
- **full event normalization**: every ``LiveServerMessage`` is parsed here and
  surfaced as a clean ``BackendEvent``. Per-turn bookkeeping (output-transcript
  accumulation, tool-call-name lookup for responses, token usage) lives on the
  ``GeminiLiveSession`` handle so the backend object stays stateless.

Config: shares the assistant backends' core keys -- ``model`` (required),
``speaker_id`` (a Gemini voice name here, default ``"Kore"``). ``api_key`` is
OPTIONAL for Gemini (unlike the OpenAI family): absent, it falls back to
``GOOGLE_API_KEY``, then to Vertex AI (``GOOGLE_CLOUD_PROJECT`` /
``GOOGLE_CLOUD_LOCATION``), then to google-genai default credential resolution
(ADC) -- so it does not use the base ``_resolve_api_key`` (which requires a key).
``language`` (or ``language_code``) sets speech-synthesis language; the worker
passes the run language through ``backend_args``.

Scope note (docs/refactor-backend-migration.md): this backend faithfully ports
the **assistant** path from ``eva.assistant.gemini_live_server`` (automatic-VAD
turn-taking). Two assistant-side deltas vs the OpenAI backend, both benign:
user-input transcripts carry no speech-start timestamp (Gemini surfaces no input
speech-boundary event, so ``TRANSCRIPT`` stream=input has ts=None downstream),
and 24 kHz role input is resampled to Gemini's 16 kHz (mulaw for a pcmu caller).
"""

from __future__ import annotations

import audioop
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from google import genai
from google.genai import types

from eva.backend.base import (
    Backend,
    BackendEvent,
    BackendEventType,
    BackendSession,
    ToolCallRequest,
    ToolCallResult,
)
from eva.backend.capabilities import BackendCapabilities
from eva.utils.audio_utils import mulaw_8k_to_pcm16_16k
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE = 24000
PCMU_SAMPLE_RATE = 8000
GEMINI_INPUT_SAMPLE_RATE = 16000  # Gemini Live expects 16 kHz PCM input
DEFAULT_VOICE = "Kore"


@dataclass
class GeminiLiveSession(BackendSession):
    """Live state for one Gemini Live session.

    Carries the genai client, the entered live-connection context manager + the
    live session, and the per-turn accumulators the normalizer needs (whether a
    model turn is in flight, accumulated output-transcript text, tool-call name
    lookup for ``send_tool_response``, latest usage, input resampler state). All
    session state, deliberately off the stateless backend object.
    """

    client: genai.Client
    conn_cm: Any
    live: Any
    in_model_turn: bool = False
    output_transcript_parts: list[str] = field(default_factory=list)
    has_function_calls: bool = False
    tool_names: dict[str, str] = field(default_factory=dict)
    usage: dict[str, int] | None = None
    resampler_state: Any = None


class GeminiLiveBackend(Backend):
    """One Gemini Live session behind the role-agnostic ``Backend`` contract.

    Construction is cheap and network-free (client + connection are created in
    ``open()``). Takes a single flat ``config`` and assembles the
    ``LiveConnectConfig`` itself -- the caller never hand-builds provider JSON.
    Recognized keys (see the module docstring for auth/language details):

    - ``model`` (required). ``speaker_id`` (default ``"Kore"``): the Gemini voice.
    - ``api_key`` (optional): else ``GOOGLE_API_KEY`` env, else Vertex AI / ADC.
    - ``language`` / ``language_code``: speech-synthesis language.
    - ``output_sample_rate`` (default 24000). ``input_format``: ``"pcm"``
      (default; role sends ``output_sample_rate`` PCM16, resampled to 16 kHz) or
      ``"pcmu"`` (telephony mulaw, converted to 16 kHz).
    - ``vad_settings``: turn-detection tunables (``silence_duration_ms``).
    - ``accent``: rejected if set (realized via ElevenLabs agent IDs).

    Vertex-only extras (for Vertex-preview Live models, e.g. ``*-live-preview``):
    - ``project`` / ``location``: Vertex project + region (else ``GOOGLE_CLOUD_*``
      / ``VERTEXAI_*`` env). A resolvable project (or ``GOOGLE_GENAI_USE_VERTEXAI``)
      routes through Vertex and ignores ``api_key``; ``"global"`` is forced to a
      region since Live requires a regional endpoint.
    - ``endpoint`` / ``api_version``: optional Vertex endpoint / API-version overrides.
    - ``function_response_scheduling``: ``"WHEN_IDLE"`` | ``"INTERRUPT"`` |
      ``"SILENT"``. Omitted by default (newer Live models reject it with 1007).
    - ``thinking_config``: dict with optional ``thinking_budget`` (int),
      ``include_thoughts`` (bool), ``thinking_level`` (str).
    """

    # api_key is optional (Vertex/ADC fallback), so it is looked up inline rather
    # than via the base _resolve_api_key (which requires a key).
    _API_KEY_ENV: ClassVar[str] = "GOOGLE_API_KEY"

    _CAPABILITIES = BackendCapabilities(
        emits_continuous_audio=True,
        supports_streaming_interruption=True,
        owns_playout_clock=False,
    )

    def __init__(self, *, config: dict[str, Any]) -> None:
        errors: list[str] = []
        self._model = self._require(config, "model", errors)
        if config.get("accent") is not None:
            errors.append("accent variants are not supported (accents are realized via ElevenLabs agents)")
        self._raise_config_errors(errors)

        self._api_key = config.get("api_key") or os.environ.get(self._API_KEY_ENV) or ""
        self._voice = config.get("speaker_id", DEFAULT_VOICE)
        self._language_code = config.get("language_code") or config.get("language")
        self._input_format: str = config.get("input_format", "pcm")
        self._output_sample_rate = int(config.get("output_sample_rate", DEFAULT_SAMPLE_RATE))
        vad = config.get("vad_settings") or {}
        self._silence_duration_ms = int(vad.get("silence_duration_ms", 200))

        # Optional Vertex endpoint / API-version overrides; when unset the SDK
        # uses its defaults. Live/S2S preview models are Vertex-only and may
        # require a specific api_version (e.g. "v1beta1").
        self._endpoint = config.get("endpoint")
        self._api_version = config.get("api_version")

        # Optional FunctionResponse scheduling ("WHEN_IDLE" | "INTERRUPT" |
        # "SILENT"). Newer Live models (e.g. gemini-3.5-flash-live-preview) do
        # NOT support a scheduling field and close the socket with 1007 if one is
        # set, so it is OMITTED by default; older models can opt back in.
        self._fc_scheduling = config.get("function_response_scheduling")

        # Vertex project/location resolution. Accept both the google-genai names
        # (GOOGLE_CLOUD_*) and the LiteLLM/Vertex names (VERTEXAI_*) so a single
        # set of credentials works for both the judge and the Live server.
        self._vertex_project = (
            config.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEXAI_PROJECT")
        )
        # Gemini Live/S2S requires a REGIONAL endpoint; "global" (fine for the
        # text judge) is not supported for bidiGenerateContent, so it is never
        # used here — an explicit region wins, otherwise fall back to a region.
        location = (
            config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEXAI_LOCATION")
        )
        if not location or location == "global":
            if location == "global":
                logger.warning(
                    "Gemini Live does not support location='global'; using 'us-central1' instead. "
                    "Set s2s_params['location'] or GOOGLE_CLOUD_LOCATION to the region the model is enabled in."
                )
            location = "us-central1"
        self._vertex_location = location

        # Thinking config: controls Gemini's internal reasoning budget. Accepts a
        # dict with optional keys "thinking_budget" (int), "include_thoughts"
        # (bool), "thinking_level" (str). Unset -> model-dependent defaults.
        self._thinking_config = self._build_thinking_config(config.get("thinking_config", {}))

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES

    @property
    def output_sample_rate(self) -> int:
        """Sample rate (Hz) of ``AUDIO_OUTPUT`` payloads (Gemini outputs 24 kHz)."""
        return self._output_sample_rate

    @property
    def input_sample_rate(self) -> int:
        """Sample rate (Hz) the role sends via ``send(audio=...)`` (converted to 16 kHz internally)."""
        return PCMU_SAMPLE_RATE if self._input_format == "pcmu" else self._output_sample_rate

    # ── Client / config assembly ──────────────────────────────────────

    @staticmethod
    def _build_thinking_config(thinking_raw: Any) -> types.ThinkingConfig:
        """Build a ``ThinkingConfig`` from the optional flat ``thinking_config`` dict."""
        if isinstance(thinking_raw, dict) and thinking_raw:
            tc_kwargs: dict[str, Any] = {}
            if "thinking_budget" in thinking_raw:
                tc_kwargs["thinking_budget"] = int(thinking_raw["thinking_budget"])
            if "include_thoughts" in thinking_raw:
                tc_kwargs["include_thoughts"] = bool(thinking_raw["include_thoughts"])
            if "thinking_level" in thinking_raw:
                tc_kwargs["thinking_level"] = thinking_raw["thinking_level"]
            logger.info(f"Thinking config: {tc_kwargs}")
            return types.ThinkingConfig(**tc_kwargs)
        return types.ThinkingConfig()

    def _create_client(self) -> genai.Client:
        """Create a google-genai Client for Vertex AI or the Developer API.

        Vertex-only models (e.g. gemini-*-live-preview) must route through
        aiplatform.googleapis.com with ``vertexai=True``; a Developer API key
        (AIza…) would send them to generativelanguage.googleapis.com/v1beta,
        where they 404. We therefore prefer Vertex whenever a project is
        resolvable (or GOOGLE_GENAI_USE_VERTEXAI is set) and ignore any
        Developer API key in that mode. Otherwise fall back to the Developer API
        key, then to google-genai default credential resolution (ADC).
        """
        flag = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        if flag is not None:
            use_vertex = flag.strip().lower() in ("1", "true", "yes")
        else:
            use_vertex = bool(self._vertex_project)

        if use_vertex:
            if not self._vertex_project:
                raise ValueError(
                    "Vertex mode requested but no project found. Set GOOGLE_CLOUD_PROJECT / "
                    "VERTEXAI_PROJECT or s2s_params['project']."
                )
            http_kwargs: dict[str, Any] = {}
            if self._endpoint:
                http_kwargs["base_url"] = f"wss://{self._endpoint}"
            if self._api_version:
                http_kwargs["api_version"] = self._api_version
            http_options = types.HttpOptions(**http_kwargs) if http_kwargs else None

            if self._api_key:
                logger.warning(
                    "Ignoring api_key in Vertex mode (Vertex uses ADC / service-account "
                    "credentials via GOOGLE_APPLICATION_CREDENTIALS)."
                )
            logger.info(
                f"Using Vertex AI (project={self._vertex_project}, location={self._vertex_location}, "
                f"api_version={self._api_version or 'sdk-default'})"
            )
            return genai.Client(
                vertexai=True,
                project=self._vertex_project,
                location=self._vertex_location,
                http_options=http_options,
            )

        if self._api_key:
            logger.info("Using Gemini Developer API key for authentication")
            return genai.Client(api_key=self._api_key)
        logger.warning("No explicit Gemini credentials; relying on google-genai default resolution")
        return genai.Client()

    def _build_live_config(self, system_prompt: str, tools: list[dict[str, Any]] | None) -> types.LiveConnectConfig:
        """Build the ``LiveConnectConfig`` for the session from flat config + role prompt/tools."""
        config_kwargs: dict[str, Any] = {
            "response_modalities": [types.Modality.AUDIO],
            "system_instruction": system_prompt,
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._voice)),
                language_code=self._language_code,
            ),
            "realtime_input_config": types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    silence_duration_ms=self._silence_duration_ms,
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
            "thinking_config": self._thinking_config,
        }
        gemini_tools = self._format_tools(tools)
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
        return types.LiveConnectConfig(**config_kwargs)

    @staticmethod
    def _json_schema_type(python_type: str) -> str:
        """Map Python/EVA type names to Gemini Schema type strings."""
        mapping = {
            "string": "STRING",
            "str": "STRING",
            "integer": "INTEGER",
            "int": "INTEGER",
            "number": "NUMBER",
            "float": "NUMBER",
            "boolean": "BOOLEAN",
            "bool": "BOOLEAN",
            "array": "ARRAY",
            "list": "ARRAY",
            "object": "OBJECT",
            "dict": "OBJECT",
        }
        return mapping.get(python_type.lower(), "STRING")

    @classmethod
    def _convert_schema_properties(cls, props: dict[str, Any]) -> dict[str, types.Schema]:
        """Recursively convert JSON-Schema property dicts to Gemini ``Schema`` objects."""
        result: dict[str, types.Schema] = {}
        for name, defn in props.items():
            if not isinstance(defn, dict):
                result[name] = types.Schema(type=types.Type.STRING)
                continue
            schema_type = cls._json_schema_type(defn.get("type", "string"))
            kwargs: dict[str, Any] = {"type": types.Type(schema_type)}
            if "description" in defn:
                kwargs["description"] = defn["description"]
            if "enum" in defn:
                kwargs["enum"] = defn["enum"]
            if schema_type == "OBJECT" and "properties" in defn:
                kwargs["properties"] = cls._convert_schema_properties(defn["properties"])
            if schema_type == "ARRAY" and "items" in defn:
                items = defn["items"]
                if isinstance(items, dict):
                    item_kwargs: dict[str, Any] = {
                        "type": types.Type(cls._json_schema_type(items.get("type", "string")))
                    }
                    if "properties" in items:
                        item_kwargs["properties"] = cls._convert_schema_properties(items["properties"])
                    kwargs["items"] = types.Schema(**item_kwargs)
                else:
                    kwargs["items"] = types.Schema(type=types.Type.STRING)
            result[name] = types.Schema(**kwargs)
        return result

    @classmethod
    def _format_tools(cls, tools: list[dict[str, Any]] | None) -> list[types.Tool] | None:
        """Translate generic ``{name, description, parameters}`` specs to Gemini tools."""
        declarations: list[types.FunctionDeclaration] = []
        for tool in tools or []:
            params = tool.get("parameters") or {}
            properties = cls._convert_schema_properties(params.get("properties", {}))
            required = params.get("required") or None
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=types.Schema(type=types.Type.OBJECT, properties=properties, required=required),
                    behavior=types.Behavior.BLOCKING,
                )
            )
        if not declarations:
            return None
        return [types.Tool(function_declarations=declarations)]

    # ── Session lifecycle ─────────────────────────────────────────────

    async def open(self, *, system_prompt: str, tools: list[dict[str, Any]] | None) -> GeminiLiveSession:
        """Connect and configure a new Gemini Live session; return its handle."""
        client = self._create_client()
        live_config = self._build_live_config(system_prompt, tools)
        conn_cm = client.aio.live.connect(model=self._model, config=live_config)
        live = await conn_cm.__aenter__()
        logger.info(f"Gemini Live session opened (model={self._model})")
        return GeminiLiveSession(client=client, conn_cm=conn_cm, live=live)

    async def send(
        self,
        session: BackendSession,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        tool_result: ToolCallResult | None = None,
    ) -> None:
        """Push audio / a text turn / a tool result to ``session`` (exactly one)."""
        provided = [x is not None for x in (audio, text, tool_result)]
        if sum(provided) != 1:
            raise ValueError("send() requires exactly one of audio, text, tool_result")
        s = self._session(session)

        if audio is not None:
            pcm_16k = self._to_gemini_input(s, audio)
            await s.live.send_realtime_input(audio=types.Blob(data=pcm_16k, mime_type="audio/pcm;rate=16000"))
            return

        if text is not None:
            await s.live.send_realtime_input(text=text)
            return

        assert tool_result is not None  # exactly-one check above
        # Only set the scheduling field when explicitly configured — newer Live
        # models reject it and close the socket with 1007.
        fr_kwargs: dict[str, Any] = {
            "id": tool_result.call_id,
            "name": s.tool_names.get(tool_result.call_id),
            "response": tool_result.result,
        }
        if self._fc_scheduling:
            fr_kwargs["scheduling"] = types.FunctionResponseScheduling[self._fc_scheduling]
        await s.live.send_tool_response(function_responses=[types.FunctionResponse(**fr_kwargs)])

    def _to_gemini_input(self, session: GeminiLiveSession, audio: bytes) -> bytes:
        """Convert role-supplied input audio to Gemini's 16 kHz PCM16."""
        if self._input_format == "pcmu":
            return mulaw_8k_to_pcm16_16k(audio)
        pcm_16k, session.resampler_state = audioop.ratecv(
            audio, 2, 1, self._output_sample_rate, GEMINI_INPUT_SAMPLE_RATE, session.resampler_state
        )
        return pcm_16k

    async def receive(self, session: BackendSession) -> AsyncIterator[BackendEvent]:
        """Yield normalized events from ``session``'s connection until it ends.

        Uses the live session's manual receive (``_receive``) rather than the
        public ``receive()`` iterator, which returns after ``turn_complete`` and
        would close the session between model turns (mirrors the old server).
        """
        s = self._session(session)
        while True:
            try:
                response = await s.live._receive()
            except Exception as e:
                logger.debug(f"Gemini Live receive ended: {e}")
                return
            if response is None:
                continue
            for be in self._map_response(s, response):
                yield be

    @staticmethod
    def _session(session: BackendSession) -> GeminiLiveSession:
        if not isinstance(session, GeminiLiveSession):
            raise TypeError(f"expected GeminiLiveSession, got {type(session).__name__}")
        return session

    def _map_response(self, session: GeminiLiveSession, response: Any) -> list[BackendEvent]:
        """Normalize one Gemini ``LiveServerMessage`` into zero or more clean ``BackendEvent``s.

        Stateful (accumulates output transcript / turn flags / usage on
        ``session``). Pure of I/O, so unit-testable with a fake session + message.
        """
        out: list[BackendEvent] = []
        sc = getattr(response, "server_content", None)

        if sc is not None:
            model_turn = getattr(sc, "model_turn", None)
            if model_turn:
                if not session.in_model_turn:
                    session.in_model_turn = True
                    session.output_transcript_parts = []
                    out.append(BackendEvent(event_type=BackendEventType.OUTPUT_TURN_STARTED))
                for part in getattr(model_turn, "parts", None) or []:
                    inline = getattr(part, "inline_data", None)
                    data = getattr(inline, "data", None) if inline is not None else None
                    if data and len(data) >= 6:
                        out.append(BackendEvent(event_type=BackendEventType.AUDIO_OUTPUT, audio=bytes(data)))

            input_tx = getattr(sc, "input_transcription", None)
            if input_tx is not None:
                text = (getattr(input_tx, "text", "") or "").strip()
                if text:
                    out.append(
                        BackendEvent(
                            event_type=BackendEventType.TRANSCRIPT,
                            transcript=text,
                            metadata={"stream": "input", "final": True},
                        )
                    )

            output_tx = getattr(sc, "output_transcription", None)
            if output_tx is not None:
                chunk = (getattr(output_tx, "text", "") or "").strip()
                if chunk:
                    session.output_transcript_parts.append(chunk)

            if getattr(sc, "interrupted", False):
                out.append(self._turn_end_event(session, interrupted=True))
                self._reset_turn(session)
            elif getattr(sc, "turn_complete", False):
                partial = " ".join(session.output_transcript_parts).strip()
                if partial:
                    out.append(
                        BackendEvent(
                            event_type=BackendEventType.TRANSCRIPT,
                            transcript=partial,
                            metadata={"stream": "output", "final": True},
                        )
                    )
                out.append(BackendEvent(event_type=BackendEventType.OUTPUT_AUDIO_DONE))
                out.append(self._turn_end_event(session, interrupted=False))
                self._reset_turn(session)

        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None:
            for fc in getattr(tool_call, "function_calls", None) or []:
                session.has_function_calls = True
                call_id = getattr(fc, "id", "") or ""
                name = getattr(fc, "name", "") or ""
                session.tool_names[call_id] = name
                out.append(
                    BackendEvent(
                        event_type=BackendEventType.TOOL_CALL_REQUEST,
                        tool_call_request=ToolCallRequest(
                            call_id=call_id,
                            name=name,
                            arguments=dict(fc.args) if getattr(fc, "args", None) else {},
                        ),
                    )
                )

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
            if prompt_tokens or completion_tokens:
                session.usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        return out

    @staticmethod
    def _turn_end_event(session: GeminiLiveSession, *, interrupted: bool) -> BackendEvent:
        return BackendEvent(
            event_type=BackendEventType.TURN_END,
            transcript=" ".join(session.output_transcript_parts).strip(),
            metadata={
                "interrupted": interrupted,
                "cancelled": False,
                "has_function_calls": session.has_function_calls,
                "usage": session.usage,
            },
        )

    @staticmethod
    def _reset_turn(session: GeminiLiveSession) -> None:
        session.in_model_turn = False
        session.output_transcript_parts = []
        session.has_function_calls = False
        session.usage = None

    async def close(self, session: BackendSession) -> None:
        """Tear down ``session``. Idempotent: safe to call more than once."""
        s = self._session(session)
        if s.conn_cm is not None:
            try:
                await s.conn_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing Gemini Live connection: {e}")
            finally:
                s.conn_cm = None
                s.live = None
