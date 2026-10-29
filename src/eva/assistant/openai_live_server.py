"""OpenAI GPT-Live assistant server implementation.

Bridges audio between a Twilio-framed WebSocket (user simulator) and OpenAI's
Live API (``client.live.connect()``, model ``gpt-live-1``).  Live is server-driven
and has no turn events, so conversation turns, interruptions and response latency
are derived by ``TurnTracker`` from transcript fragments, audio loudness and the
simulator's speech events.  Tools run behind Responses delegation: the voice model
delegates, a backend Responses model picks tools, and tool calls are executed
locally through the shared ToolExecutor.
"""

import array
import asyncio
import base64
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.openai_realtime_server import MULAW_CHUNK_DURATION_S, MULAW_CHUNK_SIZE, USER_ACTIVE_GUARD_S
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.models.agents import AgentConfig
from eva.utils.audio_utils import (
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

SESSION_START_TIMEOUT_S = 15.0
TICK_INTERVAL_S = 0.1

# GPT-Live is configured for 24 kHz 16-bit mono PCM, matching the EVA recording rate.
LIVE_SAMPLE_RATE = 24000

_FRONTEND_KEY = "frontend_instructions"
_BACKEND_KEY = "backend_instructions"


def load_live_prompts(agent_config_path: str | Path) -> tuple[str, str]:
    """Load the (frontend, backend) prompts for an agent from ``configs/agents/gpt-live/``.

    The file sits in a ``gpt-live/`` directory next to the agent YAML and shares its filename.
    """
    agent_path = Path(agent_config_path)
    live_path = agent_path.parent / "gpt-live" / agent_path.name
    if not live_path.is_file():
        raise ValueError(
            f"No GPT-Live prompt file at {live_path} (expected in a gpt-live/ directory next to {agent_path})"
        )

    with open(live_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{live_path} must be a YAML mapping with {_FRONTEND_KEY} and {_BACKEND_KEY}")

    prompts = []
    for key in (_FRONTEND_KEY, _BACKEND_KEY):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{live_path} is missing a non-empty '{key}'")
        prompts.append(value)
    return prompts[0], prompts[1]


def build_function_tools(agent: AgentConfig) -> list[dict]:
    """Convert agent tools to the flat function-tool format used in ``delegation.responses.tools``."""
    return [
        {
            "type": "function",
            "name": tool.function_name,
            "description": f"{tool.name}: {tool.description}",
            "parameters": {
                "type": "object",
                "properties": tool.get_parameter_properties(),
                "required": tool.get_required_param_names(),
            },
        }
        for tool in agent.tools
    ]


def build_session_config(
    *,
    model: str,
    frontend_instructions: str,
    backend_instructions: str,
    delegation_model: str,
    tools: list[dict],
    voice: str = "marin",
    reasoning_effort: str | None = None,
    acknowledgement_delay_ms: int | None = None,
    parallel_tool_calls: bool | None = None,
) -> dict[str, Any]:
    """Build the ``session.start`` payload: frontend voice prompt plus Responses delegation."""
    responses: dict[str, Any] = {
        "model": delegation_model,
        "instructions": backend_instructions,
        "tools": tools,
        "tool_choice": "auto",
    }
    if reasoning_effort:
        responses["reasoning"] = {"effort": reasoning_effort}
    if parallel_tool_calls is not None:
        responses["parallel_tool_calls"] = parallel_tool_calls

    delegation: dict[str, Any] = {"type": "responses", "responses": responses}
    if acknowledgement_delay_ms is not None:
        delegation["acknowledgement_delay_ms"] = acknowledgement_delay_ms

    return {
        "model": model,
        "instructions": frontend_instructions,
        "audio": {
            "format": {"type": "audio/pcm", "rate": LIVE_SAMPLE_RATE},
            "output": {"voice": voice},
        },
        "delegation": delegation,
    }


# Turn-tracker tuning. These are first guesses, to be tuned after the first real run.
LOUD_RMS_THRESHOLD = 200.0  # RMS (int16 units) above which received assistant audio counts as speech
SPEAKING_HOLD_MS = 500  # assistant counts as speaking this long after its last loud chunk or transcript delta
ASSISTANT_QUIET_FLUSH_MS = 1500  # quiet time after which an open assistant turn is flushed
USER_QUIET_FLUSH_MS = 1200  # input-transcript quiet time (after the user stopped) before the user turn is flushed
INTERRUPT_TAIL_MS = 1000  # how long an interrupted assistant turn stays open for late transcript fragments
USER_MERGE_WINDOW_MS = 2000  # a transcript-less speech start older than this is not merged into the next one

# Live's output is clocked by the input audio it receives, and the user simulator sends nothing until the
# assistant has spoken, so we fill idle input with silence (guard value is a guess; tune after a real run).
SILENCE_FEED_GUARD_S = 0.1  # no real user frame for this long -> feed silence
_SILENCE_CHUNK_B64 = base64.b64encode(b"\x00" * (LIVE_SAMPLE_RATE * 2 // 50)).decode("ascii")  # 20 ms

_MAX_LATENCY_MS = 30_000
_INTERRUPTED_SUFFIX = " [interrupted]"
_SENTENCE_END = (".", "?", "!", "…")


@dataclass
class TurnEvent:
    """A completed conversation turn derived by ``TurnTracker``."""

    kind: Literal["user", "assistant"]
    text: str
    timestamp_ms: str | None
    interrupted: bool = False


def _rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class TurnTracker:
    """Derive turns, interruptions and response latency from Live's turn-less event stream.

    Live emits continuous (silence-padded) output audio and transcript fragments with no turn
    boundaries, so "assistant is speaking" comes from audio loudness and transcript activity,
    never from audio arrival.  Transcripts trail the audio (and the user's speech stop) by up to
    about a second, so turns are flushed only after their transcript has gone quiet.  All times are
    caller-supplied wall-clock epoch milliseconds.
    """

    def __init__(self) -> None:
        self._user_parts: list[str] = []
        self._user_start_ms: int | None = None
        self._user_stop_ms: int | None = None
        self._user_first_delta_ms: int | None = None
        self._last_input_delta_ms: int | None = None
        self._user_stopped = False
        self._latency_anchor_ms: int | None = None

        self._asst_parts: list[str] = []
        self._asst_open = False
        self._asst_first_audio_ms: int | None = None
        self._asst_first_delta_ms: int | None = None
        self._last_activity_ms: int | None = None
        self._asst_interrupted_at_ms: int | None = None
        self._split_at_sentence_end = False

    def assistant_speaking(self, now_ms: int) -> bool:
        return (
            self._asst_open
            and self._last_activity_ms is not None
            and now_ms - self._last_activity_ms <= SPEAKING_HOLD_MS
        )

    def on_user_speech_start(self, now_ms: int) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        if self._asst_interrupted_at_ms is not None:
            # A second interruption: close the first interrupted turn now.
            events += self._flush_assistant(interrupted=True)
        if self._asst_open:
            if self.assistant_speaking(now_ms):
                # Keep the turn open: its transcript trails the audio, so fragments are still in flight.
                self._asst_interrupted_at_ms = now_ms
            else:
                events += self._flush_assistant(interrupted=False)
        if self._user_stopped:
            events += self._flush_user()
        # A repeated start inside one utterance keeps the original start time, but a start that never
        # produced a transcript (noise) is not carried into a much later utterance.
        stale = (
            not self._user_text()
            and self._user_stop_ms is not None
            and now_ms - self._user_stop_ms > USER_MERGE_WINDOW_MS
        )
        if self._user_start_ms is None or stale:
            self._user_start_ms = now_ms
        self._user_stopped = False
        self._latency_anchor_ms = None
        return events

    def on_user_speech_stop(self, now_ms: int) -> None:
        self._user_stopped = True
        self._user_stop_ms = now_ms
        self._latency_anchor_ms = now_ms

    def on_input_transcript(self, delta: str, now_ms: int) -> None:
        self._user_parts.append(delta)
        self._last_input_delta_ms = now_ms
        if self._user_first_delta_ms is None:
            self._user_first_delta_ms = now_ms

    def on_output_transcript(self, delta: str, now_ms: int) -> list[TurnEvent]:
        events = self._expire_interrupted(now_ms)
        self._asst_parts.append(delta)
        self._asst_open = True
        if self._asst_first_delta_ms is None:
            self._asst_first_delta_ms = now_ms
        self._last_activity_ms = now_ms
        if (
            self._split_at_sentence_end
            and self._asst_interrupted_at_ms is None
            and delta.rstrip().endswith(_SENTENCE_END)
        ):
            events += self._flush_assistant(interrupted=False)
        return events

    def on_audio_chunk(self, pcm: bytes, now_ms: int) -> float | None:
        """Return response latency in seconds the first time loud audio follows a user speech stop."""
        if _rms(pcm) < LOUD_RMS_THRESHOLD:
            return None
        self._asst_open = True
        if self._asst_first_audio_ms is None:
            self._asst_first_audio_ms = now_ms
        self._last_activity_ms = now_ms
        if self._latency_anchor_ms is None:
            return None
        latency_ms = now_ms - self._latency_anchor_ms
        self._latency_anchor_ms = None
        return latency_ms / 1000 if 0 < latency_ms < _MAX_LATENCY_MS else None

    def on_tool_call(self, now_ms: int) -> list[TurnEvent]:
        """Split pre-tool speech into its own assistant turn, without cutting a sentence in half."""
        text = "".join(self._asst_parts).strip()
        if not text or self._asst_interrupted_at_ms is not None:
            return []
        if text.endswith(_SENTENCE_END):
            return self._flush_assistant(interrupted=False)
        # The acknowledgement is still being spoken: split at the end of its current sentence.
        self._split_at_sentence_end = True
        return []

    def tick(self, now_ms: int) -> list[TurnEvent]:
        events = self._expire_interrupted(now_ms)
        if (
            self._user_stopped
            and self._last_input_delta_ms is not None
            and now_ms - self._last_input_delta_ms >= USER_QUIET_FLUSH_MS
        ):
            events += self._flush_user()
        if (
            self._asst_open
            and self._asst_interrupted_at_ms is None
            and self._last_activity_ms is not None
            and now_ms - self._last_activity_ms >= ASSISTANT_QUIET_FLUSH_MS
        ):
            events += self._flush_assistant(interrupted=False)
        return events

    def finalize(self, now_ms: int) -> list[TurnEvent]:
        events = self._flush_user() + self._flush_assistant(interrupted=self._asst_interrupted_at_ms is not None)
        events.sort(key=lambda e: int(e.timestamp_ms or 0))
        return events

    def _user_text(self) -> str:
        return "".join(self._user_parts).strip()

    def _expire_interrupted(self, now_ms: int) -> list[TurnEvent]:
        if self._asst_interrupted_at_ms is not None and now_ms - self._asst_interrupted_at_ms >= INTERRUPT_TAIL_MS:
            return self._flush_assistant(interrupted=True)
        return []

    def _flush_user(self) -> list[TurnEvent]:
        text = self._user_text()
        start = self._user_start_ms if self._user_start_ms is not None else self._user_first_delta_ms
        if not text:
            return []
        self._user_parts = []
        self._user_start_ms = None
        self._user_stop_ms = None
        self._user_first_delta_ms = None
        self._last_input_delta_ms = None
        self._user_stopped = False
        return [TurnEvent("user", text, str(start) if start is not None else None)]

    def _flush_assistant(self, interrupted: bool) -> list[TurnEvent]:
        text = "".join(self._asst_parts).strip()
        start = self._asst_first_audio_ms if self._asst_first_audio_ms is not None else self._asst_first_delta_ms
        self._asst_parts = []
        self._asst_open = False
        self._asst_first_audio_ms = None
        self._asst_first_delta_ms = None
        self._last_activity_ms = None
        self._asst_interrupted_at_ms = None
        self._split_at_sentence_end = False
        if not text:
            return []
        if interrupted:
            text += _INTERRUPTED_SUFFIX
        return [TurnEvent("assistant", text, str(start) if start is not None else None, interrupted)]


def _now_ms() -> int:
    return int(time.time() * 1000)


class OpenAILiveAssistantServer(AbstractAssistantServer):
    """Assistant server backed by the OpenAI GPT-Live API with Responses delegation.

    Exposes a local WebSocket at ``ws://localhost:{port}/ws`` using the Twilio frame format so the
    user simulator can connect as if talking to Twilio.  Audio is bridged between Twilio (8 kHz
    mulaw) and Live (24 kHz PCM16 base64).
    """

    _service_name: str = "OpenAI Live"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        s2s_params = self.pipeline_config.s2s_params or {}
        for key in ("model", "api_key", "delegation_model"):
            if not s2s_params.get(key):
                raise ValueError(f"s2s_params.{key} is required for {self._service_name}")
        self._model: str = s2s_params["model"]
        self._audio_sample_rate = LIVE_SAMPLE_RATE

        self._frontend_instructions, self._backend_instructions = load_live_prompts(self.agent_config_path)
        self._tools: list[dict] = build_function_tools(self.agent)

        self._tracker = TurnTracker()
        self._started = asyncio.Event()
        # Per-session flags so the first audio chunk / transcript is logged once (diagnosing a silent start).
        self._logged_first_audio = False
        self._logged_first_transcript = False
        # delegation id -> monotonic time its delegation.created arrived, to log how long the backend took.
        self._delegation_started: dict[str | None, float] = {}
        self._pending_calls: dict[str | None, list[tuple[str, str, dict, asyncio.Task]]] = {}
        self._stream_sid: str = ""
        self._user_speaking = False
        # Monotonic time of the last real user-audio frame appended; guards against padding the user
        # track mid-utterance (see USER_ACTIVE_GUARD_S).
        self._last_user_audio_mono: float = 0.0

    def _build_session_config(self) -> dict[str, Any]:
        s2s = self.pipeline_config.s2s_params
        return build_session_config(
            model=self._model,
            frontend_instructions=self._frontend_instructions,
            backend_instructions=self._backend_instructions,
            delegation_model=s2s["delegation_model"],
            tools=self._tools,
            voice=s2s.get("voice", "marin"),
            reasoning_effort=s2s.get("delegation_reasoning_effort"),
            acknowledgement_delay_ms=s2s.get("acknowledgement_delay_ms"),
            parallel_tool_calls=self.pipeline_config.parallel_tool_calls,
        )

    async def start(self) -> None:
        """Start the FastAPI WebSocket server."""
        if self._running:
            logger.warning("Server already running")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"{self._service_name} server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the OpenAI Live server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        logger.info(f"{self._service_name} server stopped on port {self.port}")

    # ── Session ───────────────────────────────────────────────────────

    async def _open_session(self, conn: Any) -> None:
        """Send ``session.start``; the server confirms with ``session.started``."""
        await conn.session.start(session=self._build_session_config())

    async def _greet_when_started(self, conn: Any) -> None:
        """Have the model speak the greeting once the session is up (Live never speaks first)."""
        try:
            await asyncio.wait_for(self._started.wait(), timeout=SESSION_START_TIMEOUT_S)
        except TimeoutError:
            # Raising (not just logging) lets _handle_session end the run visibly instead of leaving a deaf model.
            raise RuntimeError(
                f"{self._service_name} did not send session.started within {SESSION_START_TIMEOUT_S:.0f}s"
            ) from None
        await conn.session.commentary.append(
            content=f"The call has just connected. Greet the caller now by saying: '{self.initial_message}'",
            delegation_id=None,
        )
        logger.info(f"{self._service_name} greeting sent: {self.initial_message!r}")

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Handle a single WebSocket session: connect to Live, then bridge audio and events until one side ends."""
        logger.info(f"Client connected to {self._service_name} server")

        self._started.clear()
        self._logged_first_audio = False
        self._logged_first_transcript = False
        self._stream_sid = self.conversation_id
        self._user_speaking = False

        client = AsyncOpenAI(api_key=self.pipeline_config.s2s_params["api_key"])

        try:
            logger.info(f"Starting {self._service_name} session (model={self._model})")
            async with client.live.connect() as conn:
                await self._open_session(conn)

                audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()
                # Set when the greeting fails (session never started), so the whole session ends visibly.
                abort = asyncio.Event()
                tasks = [
                    asyncio.create_task(self._forward_user_audio(websocket, conn)),
                    asyncio.create_task(self._process_live_events(conn, audio_output_queue)),
                    asyncio.create_task(self._pace_audio_output(websocket, audio_output_queue)),
                    asyncio.create_task(self._tick_loop()),
                    asyncio.create_task(self._feed_silence_to_live(conn)),
                    asyncio.create_task(abort.wait()),
                ]
                # The greeting finishes quickly, so it is kept out of the wait set below.
                greeting_task = asyncio.create_task(self._greet_when_started(conn))
                greeting_task.add_done_callback(lambda t: abort.set() if not t.cancelled() and t.exception() else None)

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in [*pending, greeting_task]:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Session task failed: {e}")

                for task in done:
                    if task.exception():
                        logger.error(f"Session task failed: {task.exception()}")

        except Exception as e:
            logger.error(f"{self._service_name} session error: {e}", exc_info=True)
        finally:
            self._emit(self._tracker.finalize(_now_ms()))
            logger.info(f"Client disconnected from {self._service_name} server")

    async def _tick_loop(self) -> None:
        """Periodically flush assistant turns that have gone quiet."""
        try:
            while True:
                await asyncio.sleep(TICK_INTERVAL_S)
                self._emit(self._tracker.tick(_now_ms()))
        except asyncio.CancelledError:
            pass

    async def _feed_silence_to_live(self, conn: Any) -> None:
        """Send real-time silence to Live whenever no user audio is arriving.

        Without input audio Live's clock does not advance: it produces no output, so the greeting never plays and
        the simulator (which only sends silence after hearing the assistant) never starts streaming.
        """
        next_send = time.monotonic()
        try:
            while True:
                next_send += MULAW_CHUNK_DURATION_S
                now = time.monotonic()
                if next_send < now - 0.1:
                    next_send = now  # fell behind (e.g. event-loop stall): do not burst to catch up
                await asyncio.sleep(max(0.0, next_send - now))
                if not self._started.is_set():
                    continue
                if time.monotonic() - self._last_user_audio_mono <= SILENCE_FEED_GUARD_S:
                    continue
                await conn.session.input_audio.append(audio=_SILENCE_CHUNK_B64)
        except asyncio.CancelledError:
            pass

    # ── Audio output pacer (Live -> Twilio WS at real-time rate) ──────

    async def _pace_audio_output(self, websocket: WebSocket, audio_output_queue: asyncio.Queue[bytes]) -> None:
        """Drain audio_output_queue and forward chunks at real-time rate."""
        next_send_time = time.monotonic()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                twilio_msg = create_twilio_media_message(self._stream_sid, chunk)
                try:
                    await websocket.send_text(twilio_msg)
                except Exception as e:
                    logger.error(f"Error sending audio to Twilio WS: {e}")
                    return

                now = time.monotonic()
                if next_send_time <= now:
                    next_send_time = now
                next_send_time += MULAW_CHUNK_DURATION_S
                sleep_duration = next_send_time - time.monotonic()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            pass

    # ── User audio forwarding (Twilio WS -> Live) ─────────────────────

    @staticmethod
    def _event_ms(data: dict) -> int:
        """Wall-clock ms from a simulator event, falling back to now."""
        return int(float(data.get("timestamp_ms") or _now_ms()))

    async def _forward_user_audio(self, websocket: WebSocket, conn: Any) -> None:
        """Read Twilio media frames, record them, and forward audio to Live once the session has started."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                event_type = data.get("event")

                if event_type == "start":
                    self._stream_sid = data.get("start", {}).get("streamSid", self.conversation_id)
                    logger.debug(f"Twilio stream started: streamSid={self._stream_sid}")
                    continue

                if event_type == "stop":
                    logger.debug("Twilio stream stopped")
                    break

                if event_type == "user_speech_start":
                    self._user_speaking = True
                    self._emit(self._tracker.on_user_speech_start(self._event_ms(data)))
                    continue

                if event_type == "user_speech_stop":
                    self._user_speaking = False
                    # The bridge stamps this event with its monotonic loop time, not epoch ms, so use receive
                    # time (as the other S2S servers do for their own stop events).
                    self._tracker.on_user_speech_stop(_now_ms())
                    continue

                if event_type != "media":
                    continue

                mulaw_bytes = parse_twilio_media_message(raw)
                if mulaw_bytes is None:
                    continue

                pcm16_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)

                if not self._tracker.assistant_speaking(_now_ms()):
                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                self.user_audio_buffer.extend(pcm16_24k)
                self._last_user_audio_mono = time.monotonic()

                # Audio sent before session.started raced in the spike, so it is only recorded until then.
                if self._started.is_set():
                    await conn.session.input_audio.append(audio=base64.b64encode(pcm16_24k).decode("ascii"))

        except WebSocketDisconnect:
            logger.debug("Twilio WebSocket disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error forwarding user audio: {e}", exc_info=True)

    # ── Live event processing ─────────────────────────────────────────

    async def _process_live_events(self, conn: Any, audio_output_queue: asyncio.Queue[bytes]) -> None:
        """Process events from the Live connection."""
        try:
            async for event in conn:
                try:
                    await self._handle_live_event(event, conn, audio_output_queue)
                except Exception as e:
                    logger.error(f"Error handling event {getattr(event, 'type', '?')}: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in Live event loop: {e}", exc_info=True)

    async def _handle_live_event(self, event: Any, conn: Any, audio_output_queue: asyncio.Queue[bytes]) -> None:
        """Dispatch a single Live server event."""
        event_type = getattr(event, "type", "")

        match event_type:
            case "session.started":
                logger.info(f"{self._service_name} session started")
                self._started.set()
                # Marks the run as delegation-capable so the delegation_rate metric can tell "no delegations"
                # from "framework without delegation".
                if self._fw_log:
                    self._fw_log.write("delegation_session", {"frame": "delegation_session"})

            case "session.output_audio.delta":
                await self._on_audio_delta(event, audio_output_queue)

            case "session.input_transcript.delta":
                self._tracker.on_input_transcript(getattr(event, "delta", "") or "", _now_ms())

            case "session.commentary.appended":
                logger.info(f"{self._service_name} acknowledged the greeting commentary")

            case "session.output_transcript.delta":
                if not self._logged_first_transcript:
                    self._logged_first_transcript = True
                    logger.info(f"First assistant transcript delta: {getattr(event, 'delta', '')!r}")
                self._emit(self._tracker.on_output_transcript(getattr(event, "delta", "") or "", _now_ms()))

            case "response.event":
                inner = getattr(event, "event", None) or {}
                item = inner.get("item") or {}
                if inner.get("type") == "response.output_item.done" and item.get("type") == "function_call":
                    # Whatever was said before the tool call is its own assistant turn.
                    self._emit(self._tracker.on_tool_call(_now_ms()))
                await self._on_response_event(event, conn)

            case "session.delegation.created":
                delegation = getattr(event, "delegation", None)
                logger.debug(f"Delegation created: {delegation}")
                self._delegation_started[getattr(delegation, "id", None)] = time.monotonic()
                if self._fw_log:
                    self._fw_log.write(
                        "delegation",
                        {
                            "frame": "delegation",
                            "delegation_id": getattr(delegation, "id", None),
                            "response_id": getattr(delegation, "response_id", None),
                        },
                    )

            case "error":
                logger.error(f"{self._service_name} error: {event}")

            case _:
                logger.debug(f"Unhandled {self._service_name} event: {event_type}")

    async def _on_audio_delta(self, event: Any, audio_output_queue: asyncio.Queue[bytes]) -> None:
        """Handle an output audio chunk: record it, measure latency, and queue it for Twilio.

        Live streams audio continuously (silence when idle), so every chunk is recorded, but only speech
        (plus a short hold, so pauses inside speech do not split it) is forwarded: the simulator would
        otherwise see one unbroken assistant span. Speech onset is detected from loudness, not from arrival.
        """
        delta_b64 = getattr(event, "delta", "") or ""
        if not delta_b64:
            return
        pcm16_bytes = base64.b64decode(delta_b64)
        if not self._logged_first_audio:
            self._logged_first_audio = True
            logger.info(f"First assistant audio chunk received ({len(pcm16_bytes)} bytes)")

        latency_s = self._tracker.on_audio_chunk(pcm16_bytes, _now_ms())
        if latency_s is not None and self._metrics_log:
            self._metrics_log.write_latency("model_response", latency_s, self._model)

        # Skip the pad if the user track is actively receiving audio (flag may be stale under
        # jitter) — padding then would inject a mid-utterance chop.
        user_recently_active = (time.monotonic() - self._last_user_audio_mono) <= USER_ACTIVE_GUARD_S
        if not self._user_speaking and not user_recently_active:
            sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
        self.assistant_audio_buffer.extend(pcm16_bytes)

        if not self._tracker.assistant_speaking(_now_ms()):
            return

        try:
            mulaw_bytes = pcm16_24k_to_mulaw_8k(pcm16_bytes)
            for offset in range(0, len(mulaw_bytes), MULAW_CHUNK_SIZE):
                await audio_output_queue.put(mulaw_bytes[offset : offset + MULAW_CHUNK_SIZE])
        except Exception as e:
            logger.error(f"Error converting audio for output queue: {e}")

    def _emit(self, events: list[TurnEvent]) -> None:
        """Write completed turns to the audit log and framework log."""
        for turn in events:
            if not turn.text:
                continue
            logger.info(f"Turn written ({turn.kind}{', interrupted' if turn.interrupted else ''}): {turn.text[:120]!r}")
            if turn.kind == "user":
                if self._fw_log and turn.timestamp_ms is not None:
                    self._fw_log.turn_start(timestamp_ms=int(turn.timestamp_ms))
                self.audit_log.append_user_input(turn.text, timestamp_ms=turn.timestamp_ms)
            else:
                self.audit_log.append_assistant_output(turn.text, timestamp_ms=turn.timestamp_ms)
                if self._fw_log:
                    self._fw_log.llm_response(turn.text)
                    if turn.interrupted:
                        self._fw_log.s2s_transcript(turn.text)
                    self._fw_log.turn_end(was_interrupted=turn.interrupted)

    async def _on_response_event(self, event: Any, conn: Any) -> None:
        """Handle a delegated Responses event: collect tool calls, answer them when the response completes.

        Tool calls start executing as soon as their ``function_call`` item is done.  When the backend
        response completes, the outputs are sent in call order followed by a single ``response.create``
        so the backend continues once, even with parallel tool calls.
        """
        inner = event.event or {}
        delegation_id = getattr(event, "delegation_id", None)
        inner_type = inner.get("type")

        if inner_type == "response.output_item.done":
            item = inner.get("item") or {}
            if item.get("type") == "function_call":
                self._start_tool_call(delegation_id, item)
        elif inner_type == "response.output_text.done":
            started = self._delegation_started.pop(delegation_id, None)
            since = f", {time.monotonic() - started:.1f}s since delegation created" if started is not None else ""
            logger.info(f"Backend answer (delegation {delegation_id}{since}): {(inner.get('text') or '')[:300]!r}")
        elif inner_type == "response.completed":
            response = inner.get("response") or {}
            usage = response.get("usage") or {}
            logger.info(
                f"Backend response {response.get('status')} (delegation {delegation_id}): "
                f"input_tokens={usage.get('input_tokens')}, output_tokens={usage.get('output_tokens')}"
            )
            await self._answer_tool_calls(delegation_id, conn)
        elif inner_type in ("response.failed", "response.incomplete"):
            self._drop_pending_calls(delegation_id, inner_type)

    def _drop_pending_calls(self, delegation_id: str | None, reason: str) -> None:
        """The backend response will not continue, so its started tool calls have nobody to answer to."""
        pending = self._pending_calls.pop(delegation_id, [])
        for _, _, _, task in pending:
            task.cancel()
        logger.error(f"Backend {reason} for delegation {delegation_id}; dropped {len(pending)} pending tool call(s)")

    def _start_tool_call(self, delegation_id: str | None, item: dict) -> None:
        name = item.get("name", "")
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        logger.info(f"Tool call: {name}({json.dumps(arguments, ensure_ascii=False)})")
        task = asyncio.create_task(self.execute_tool(name, arguments))
        self._pending_calls.setdefault(delegation_id, []).append((item.get("call_id", ""), name, arguments, task))

    async def _answer_tool_calls(self, delegation_id: str | None, conn: Any) -> None:
        pending = self._pending_calls.pop(delegation_id, [])
        for call_id, name, arguments, task in pending:
            try:
                result = await task
            except Exception as e:
                # The model must still get an output for every call, or the backend waits forever.
                logger.error(f"Tool {name} failed: {e}", exc_info=True)
                result = {"error": f"Tool {name} failed: {e}"}
            if self._fw_log:
                self._fw_log.write(
                    "tool_call",
                    {"frame": "tool_call", "tool_name": name, "arguments": arguments, "result": result},
                )
            try:
                await conn.response.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send output for {name} ({call_id}): {e}", exc_info=True)
        if pending:
            await conn.response.create()
