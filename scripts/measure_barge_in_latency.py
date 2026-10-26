"""Controlled measurement of the barge-in reaction path, end to end.

The question: can the caller go from "the assistant is mid-sentence" to "the caller is
audibly interrupting" in roughly human reaction time, with a pre-rendered opener masking
the time it takes to generate and synthesize what the caller actually says — and with no
audible gap between the opener and that content?

The path under test:

    one assistant utterance, pre-rendered with Cartesia and streamed in real time
      -> the cascade simulator's streaming STT (ScribeStreamingSTT, ElevenLabs Scribe realtime)
      -> on the Nth transcript delta: OpenAI Decisions API (choice: interrupt / keep listening)
      -> interrupt: the pre-rendered opener starts playing immediately
      -> the caller's reply (streaming LLM -> Cartesia) plays when the opener ends

Several reply STRATEGIES are compared in one invocation; they differ only in *when* the
reply pipeline starts and whether requests are hedged — see STRATEGIES. Every connection
is opened and warmed before the clock starts and torn down after it stops.

Reply timing does not depend on which opener plays, so every run is scored against every
opener: the caller track is replayed once per opener and saved as its own recording.

The recordings are a replay of the real-time playout, not an idealised splice: the opener
starts at the moment the verdict arrived, and each content chunk at max(previous audio end,
that chunk's arrival). The headline GAP is measured from that replayed track itself — the
silence between the opener's last audible sample and the content's first.

Reaction time is measured against Cartesia word timestamps for the assistant turn: from
the end of the last word the STT had transcribed to the opener starting.

All knobs are module constants below — edit them in place, there are no CLI arguments.

    python scripts/measure_barge_in_latency.py
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import json
import os
import re
import statistics
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv
from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeEvents, ScribeRealtime

from eva.models.config import CascadeSimulatorConfig
from eva.user_simulator.cascade.stt import INPUT_SAMPLE_RATE, ScribeStreamingSTT
from eva.user_simulator.cascade.tts import CALLER_SPEED, TTS_SAMPLE_RATE, CartesiaStreamingTTS

# ══════════════════════════════════════════════════════════════════════════════════
#  KNOBS — edit these
# ══════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Strategy:
    """When the reply pipeline starts, what is hedged, and which reply model writes it."""

    start: str
    """after_decision | trigger | rolling | threshold — see STRATEGIES."""
    speculate_tts: bool = True
    """False: speculation produces text only; TTS starts on the chosen text after the verdict."""
    hedge_decision: bool = False
    hedge_reply: bool = False
    """Hedge the LLM and TTS requests too."""
    response_model: str = "gpt-5.6-luna"
    reasoning_effort: str | None = "none"
    """None omits the parameter (non-reasoning models reject it)."""
    service_tier: str = "default"
    keep_newest: bool = False
    """When ROLLING_MAX_IN_FLIGHT is reached, cancel the oldest speculation instead of skipping the new one."""
    max_stale_deltas: int | None = None
    """Only consider speculations at most this many deltas older than the trigger; among those,
    the furthest along wins. None: any speculation, furthest along then freshest."""
    tts_on_clause: bool = False
    """Send the first clause to TTS as soon as it is written, then the rest of the sentence."""


STRATEGIES = {
    "sequential": Strategy("after_decision"),
    "rolling_hedged": Strategy("rolling", hedge_decision=True, hedge_reply=True),
    "rolling_text": Strategy("rolling", speculate_tts=False, hedge_decision=True),
    "threshold_text": Strategy("threshold", speculate_tts=False, hedge_decision=True),
    "parallel_priority": Strategy("trigger", hedge_decision=True, service_tier="priority"),
    "fresh_text": Strategy("rolling", speculate_tts=False, hedge_decision=True, keep_newest=True, max_stale_deltas=0),
    "fresh_text_clause": Strategy(
        "rolling", speculate_tts=False, hedge_decision=True, keep_newest=True, max_stale_deltas=0, tts_on_clause=True
    ),
    "fresh1_text_clause": Strategy(
        "rolling", speculate_tts=False, hedge_decision=True, keep_newest=True, max_stale_deltas=1, tts_on_clause=True
    ),
}
"""Reply strategies to compare, each run ITERATIONS_PER_STRATEGY times.

Every strategy behaves as a live caller would: a decision runs on every transcript
delta, and none of them knows where the trigger is. Deltas before TRIGGER_ON_DELTA are
treated as "keep listening" (the model's real choice is recorded in delta_p); the
trigger delta's verdict commits the interruption. So the cost recorded per sample is
the whole job — listen, decide on every delta, then speak. `start`:

  after_decision  reply pipeline starts only after a committing verdict (the original design)
  trigger         a reply starts alongside every delta's decision, and is cancelled when
                  that decision says keep listening
  rolling         a speculative reply starts on every delta (at most ROLLING_MAX_IN_FLIGHT)
  threshold       a speculative reply starts on a delta whose P(interrupt) >= SPECULATE_P;
                  falls back to `trigger` behaviour at the trigger if none crossed it

On the verdict, the freshest reply that is furthest along (audio arriving > text done >
generating) is used and the rest are cancelled.
"""

ITERATIONS_PER_STRATEGY = 5

RUN_ONLY = ["fresh_text_clause"]

STT_BACKENDS = ["production"]
"""Speech-to-text paths to compare, interleaved per run so session drift hits each equally.

  production    the simulator's ScribeStreamingSTT, fed 8kHz mu-law as the bridge delivers it
                (it decodes and upsamples to 16kHz PCM16 before Scribe)
  scribe_pcm16  the same SDK and model fed the original 16kHz PCM16: more audio detail than the
                bridge delivers, so an upper bound rather than a production option

Measured on one trigger (p50 STT lag): sending Scribe the mu-law directly 540ms, the same audio
as 16kHz PCM16 422ms, clean 16kHz PCM16 401ms. A LiveKit-plugin path (16kHz) was ~101 vs ~81ms
for the SDK on another trigger. Format was the difference, so production sends PCM16.
"""
"""Strategy names to run; empty runs all of STRATEGIES. A subset writes to OUTPUT_DIR/subset/
so it never overwrites a full run's samples.json."""

OPENERS = ["Sorry—", "Wait, sorry—", "Sorry, sorry—", "Sorry, hang on a sec—"]
"""Pre-rendered latency masks, shortest to longest. Every run is scored against each."""

GAP_TARGET_MS = 100.0
"""Audible gap at or under which a run counts as seamless."""

HEDGE_FANOUT = 2
"""Identical requests per hedged call; the first to answer wins."""

ROLLING_MAX_IN_FLIGHT = 3
"""Speculative replies allowed in flight at once. A delta arriving when full starts none.

Skipping the newest rather than cancelling the oldest matters: the oldest is the one
closest to finishing, and cancelling it means that when deltas arrive faster than a
reply takes to write, no speculation ever completes.
"""

CLAUSE_MIN_WORDS = 3
"""tts_on_clause: a clause break counts only after this many words, so TTS gets enough to phrase."""

CLAUSE_BREAKS = (",", "—", "–", ";", ":")

SPECULATE_P = 0.1
"""threshold strategies: P(interrupt) at or above which a delta starts a speculative reply."""

SIMULATOR = CascadeSimulatorConfig()
"""The cascade simulator's config: STT model, TTS model and voices come from here, so the
experiment hears and speaks through the same components a benchmark run uses. Caller speed is
fixed in the TTS class (CALLER_SPEED)."""

SKIP_CONTENT_LEADING_SILENCE = True
"""Drop quiet samples at the very start of the content stream as they arrive.

Causal — a real-time player can do the same — so it does not flatter the measurement.
"""

DECISION_MODEL = "gpt-6-luna"
"""Decisions API model. Only gpt-6-luna is supported at launch."""

FORCE_INTERRUPT = True
"""Proceed with the interruption whatever the model chooses; choice and P are recorded."""

BUDGET_MS = 350.0
"""Human reaction time: end of the last heard word -> caller audio out. Informational."""

ASSISTANT_GAIN = 0.33
"""Assistant channel volume in the saved recordings, so the caller is easy to hear."""

# ── Incoming audio: one rendered assistant utterance ────────────────────────────

ASSISTANT_TURN_TEXT = (
    "Understood, you'd like to update your professional liability insurance. That requires "
    "provider verification first. Please give me your ten-digit national provider number, the "
    "code for your primary facility, and your four-digit personal identification number."
)
"""Rendered once per invocation during setup and reused for every iteration."""

ASSISTANT_VOICE_ID = SIMULATOR.male_voice

TRIGGER_ON_DELTA = 5
"""Fire the decision on this transcript delta (1-based)."""

FEED_CHUNK_MS = 20
"""Audio pushed to STT per step, in real time. Also the polling resolution for deltas."""

STT_WARMUP_MS = 1000
"""Silence fed before the clock starts, so the STT websocket is connected and streaming."""

LANGUAGE = "en"
STT_MODEL = SIMULATOR.stt_model
"""Empty uses the provider default from stt_livekit.DEFAULT_MODELS."""

CARTESIA_MODEL = SIMULATOR.tts_model
VOICE_ID = SIMULATOR.female_voice

SILENCE_RMS = 200
"""PCM16 RMS per 10ms window below which audio counts as silence (trimming and GAP)."""

OUTPUT_DIR = "output/barge_in_latency"
"""<strategy>/opener<i>_run<k>.wav and samples.json land here."""

# ══════════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════════

CALLER_SAMPLE_RATE = TTS_SAMPLE_RATE
"""All audio in this script is PCM16 mono at the caller TTS rate; STT input is converted on send."""
BYTES_PER_SECOND = CALLER_SAMPLE_RATE * 2  # PCM16 mono
WINDOW_MS = 10
OPENAI_URL = "https://api.openai.com/v1"
CARTESIA_SSE_URL = "https://api.cartesia.ai/tts/sse"
CARTESIA_VERSION = "2024-06-10"
"""Only used to render the assistant turn with word timestamps, which CartesiaStreamingTTS does not expose."""
SENTENCE_END = (".", "?", "!")
KEEPALIVE = httpx.Limits(keepalive_expiry=300.0, max_keepalive_connections=20)
"""httpx's 5s default would expire the pool during each iteration's idle stretch."""

COST = {"decision_calls": 0, "llm_calls": 0, "tts_calls": 0, "tts_chars": 0}
IN_FLIGHT = {"decisions": 0, "llm_streams": 0}
"""OpenAI requests currently open, for diagnosing what the trigger decision competes with."""
"""Billable requests in the current iteration, hedged duplicates and cancelled speculation included."""

LEADING_FILLER = re.compile(
    r"^\W*(?:(?:sorry|wait|hang on|hold on|excuse me|one sec(?:ond)?|a sec(?:ond)?)\W+)*", re.IGNORECASE
)
"""Interjections the content must not open with: the opener already said them."""

GOAL_BLOCK = """GOAL: Update your professional liability insurance on file before your credentialing deadline.
MUST HAVE:
- The agent confirms the new policy is recorded on your provider profile.
- You know whether anything else is needed to stay credentialed.
VERIFICATION DETAILS YOU HAVE READY: national provider number 4827159360, primary facility
code MRC-07, personal identification number 2291.
HOW THEY EVALUATE OPTIONS: You are between patients and short on time. You already have
every verification detail in hand and want to give them the moment you know what is asked.
RESOLVED WHEN: The agent confirms the updated policy is on file."""

DECISION_INSTRUCTIONS = """You are the caller on a live phone call with a support agent, and the agent \
is speaking right now; the final line of the transcript is still in progress. What should the \
caller do at this moment?"""

DECISION_CHOICES = [
    {
        "value": "interrupt",
        "description": "Cut in now: the caller has heard enough to have a response, question or "
        "correction ready, needs to redirect the agent, or already has what they were waiting for.",
    },
    {
        "value": "keep_listening",
        "description": "Let the agent continue: they have said only a few words, or the caller would only be agreeing.",
    },
]

RESPONSE_PROMPT = """You are the caller in a live phone call with a support agent. You are cutting the \
agent off mid-sentence, and you have ALREADY said a quick apology like "Sorry—" out loud to get \
their attention.

<user_goal>
{goal}
</user_goal>

<heard_so_far>
Agent: {transcript} [CURRENTLY SPEAKING]
</heard_so_far>

Write only the one short sentence you say right after that apology. Do not begin with an apology, \
"wait", or any other interjection: that has already been said. Spoken words only, no quotes."""


# ══════════════════════════════════════════════════════════════════════════════════
#  Samples
# ══════════════════════════════════════════════════════════════════════════════════


@dataclass
class Opener:
    """A pre-rendered, silence-trimmed opener."""

    phrase: str
    pcm: bytes
    raw_ms: float
    """Duration as rendered, before trimming."""

    @property
    def ms(self) -> float:
        """Cover time its playout buys."""
        return pcm_duration_ms(len(self.pcm))


@dataclass
class Reply:
    """One run of the reply pipeline (LLM sentence -> TTS), started at some delta."""

    transcript: str
    started_at_ms: float
    delta: int = 0
    """Transcript delta this reply was written from."""
    partial: str = ""
    """Text so far while generating."""
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    text: str = ""
    raw_text: str = ""
    gen_first_token_ms: float = 0.0
    """Reply start -> first content token."""
    gen_first_sentence_ms: float = 0.0
    """Reply start -> first sentence complete (TTS request sent)."""
    tts_ttfa_ms: float = 0.0
    """TTS request sent -> first audio byte."""
    chunks: list[tuple[float, bytes]] = field(default_factory=list)
    """(audio-clock ms arrived, PCM16) for each TTS chunk."""
    task: asyncio.Task | None = None

    @property
    def progress(self) -> int:
        """How far along: 2 audio arriving, 1 text done, 0 still generating."""
        return 2 if self.chunks else 1 if self.text else 0


@dataclass
class Sample:
    """One measured interruption. All *_at_ms are on the audio clock (0 = turn starts)."""

    strategy: str
    index: int
    stt_backend: str = ""
    ok: bool = False
    transcript: str = ""
    choice: str = ""
    p_interrupt: float = 0.0
    last_word_end_at_ms: float = 0.0
    """End of the last transcribed word in the assistant audio, from Cartesia timestamps."""
    trigger_at_ms: float = 0.0
    """Triggering transcript delta became readable."""
    opener_at_ms: float = 0.0
    """Verdict arrived; opener playout starts."""
    decision_ms: float = 0.0
    reply_started_at_ms: float = 0.0
    stale_words: int = 0
    """Words between starting_word and latest_word (see missed_words)."""
    """Words the trigger transcript had that the chosen reply's transcript did not."""
    gen_first_token_ms: float = 0.0
    gen_first_sentence_ms: float = 0.0
    tts_ttfa_ms: float = 0.0
    content_first_audio_at_ms: float = 0.0
    raw_content: str = ""
    content: str = ""
    gaps_ms: dict[str, float] = field(default_factory=dict)
    """Opener phrase -> audible gap in the replayed track."""
    replies_started: int = 0
    first_speculation_delta: int = 0
    """Delta whose transcript the first speculative reply used; 0 if none."""
    delta_p: list[tuple[int, float]] = field(default_factory=list)
    latest_word: str = ""
    """Last assistant word whose audio had fully played before the opener started."""
    starting_word: str = ""
    """Last word of the transcript the chosen reply was written from."""
    missed_words: str = ""
    """Words between starting and latest: heard, but not in the reply's context."""
    reply_transcript: str = ""
    """Transcript the chosen reply was written from."""
    source_end_at_ms: float = 0.0
    """End of that transcript's last word in the assistant audio."""
    source_age_ms: float = 0.0
    """Shortest opener: when the reply becomes audible, how old its source speech is."""
    early_interrupt_votes: int = 0
    """Pre-trigger deltas where the model itself chose interrupt (overridden to keep listening)."""
    deltas_listened: int = 0
    """threshold strategies: (delta, P(interrupt)) for every per-delta decision."""
    cost: dict[str, int] = field(default_factory=dict)
    in_flight_at_trigger: dict[str, int] = field(default_factory=dict)
    """OpenAI requests already open when the trigger decision was sent."""
    pool_before: int = 0
    pool_after: int = 0
    """OpenAI pool connections before/after the trigger decision; growth = handshakes in the clock."""
    error: str = ""

    @property
    def stt_lag_ms(self) -> float:
        """End of the last heard word -> the transcript containing it was readable."""
        return self.trigger_at_ms - self.last_word_end_at_ms

    @property
    def reaction_ms(self) -> float:
        """End of the last heard word -> opener audible. What a listener perceives."""
        return self.opener_at_ms - self.last_word_end_at_ms

    @property
    def content_ready_ms(self) -> float:
        """Opener start -> first content audio. Negative when it was ready before the verdict."""
        return self.content_first_audio_at_ms - self.opener_at_ms


# ══════════════════════════════════════════════════════════════════════════════════
#  Audio helpers
# ══════════════════════════════════════════════════════════════════════════════════


def pcm_duration_ms(num_bytes: int) -> float:
    """Duration of raw PCM16 mono at the caller sample rate."""
    return num_bytes / BYTES_PER_SECOND * 1000.0


def ms_to_bytes(ms: float) -> int:
    """Byte offset of a time in PCM16 mono, aligned to a sample."""
    offset = int(ms / 1000.0 * BYTES_PER_SECOND)
    return offset - offset % 2


def loud_windows(pcm: bytes) -> list[int]:
    """Byte offsets of the 10ms windows at or above SILENCE_RMS."""
    window = ms_to_bytes(WINDOW_MS)
    return [i for i in range(0, len(pcm), window) if audioop.rms(pcm[i : i + window], 2) >= SILENCE_RMS]


def trim_silence(pcm: bytes) -> bytes:
    """Drop leading and trailing quiet windows."""
    loud = loud_windows(pcm)
    return pcm[loud[0] : loud[-1] + ms_to_bytes(WINDOW_MS)] if loud else pcm


def place(track: bytearray, pcm: bytes, at_ms: float) -> None:
    """Overwrite pcm into a track at a time offset, growing the track if needed."""
    start = ms_to_bytes(at_ms)
    if len(track) < start + len(pcm):
        track.extend(b"\x00" * (start + len(pcm) - len(track)))
    track[start : start + len(pcm)] = pcm


def write_stereo(path: Path, left: bytes, right: bytes) -> None:
    """Interleave two PCM16 mono tracks into a stereo WAV, padding the shorter."""
    length = max(len(left), len(right))
    left = audioop.mul(left, 2, ASSISTANT_GAIN).ljust(length, b"\x00")
    right = right.ljust(length, b"\x00")
    stereo = audioop.add(audioop.tostereo(left, 2, 1, 0), audioop.tostereo(right, 2, 0, 1), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(CALLER_SAMPLE_RATE)
        wav.writeframes(stereo)


def replay_playout(opener_at_ms: float, opener: Opener, chunks: list[tuple[float, bytes]]) -> tuple[bytes, float]:
    """Rebuild the caller track as a real-time player would emit it; returns (track, gap).

    The opener starts when the verdict arrived. Each content chunk starts at the later of
    the previous audio ending and the chunk arriving. The gap is then read off the track
    itself: last audible opener sample to first audible content sample.
    """
    track = bytearray()
    place(track, opener.pcm, opener_at_ms)
    opener_end = len(track)
    cursor = opener_at_ms + opener.ms
    leading = SKIP_CONTENT_LEADING_SILENCE
    for arrived_at, chunk in chunks:
        if leading:
            # Causal: only samples that have already arrived are inspected.
            loud = loud_windows(chunk)
            if not loud:
                continue
            chunk, leading = chunk[loud[0] :], False
        start = max(cursor, arrived_at)
        place(track, chunk, start)
        cursor = start + pcm_duration_ms(len(chunk))

    opener_loud = loud_windows(bytes(track[:opener_end]))
    content_loud = loud_windows(bytes(track[opener_end:]))
    if not opener_loud or not content_loud:
        return bytes(track), float("nan")
    window = ms_to_bytes(WINDOW_MS)
    gap_bytes = opener_end + content_loud[0] - (opener_loud[-1] + window)
    return bytes(track), pcm_duration_ms(max(0, gap_bytes))


# ══════════════════════════════════════════════════════════════════════════════════
#  Hedging
# ══════════════════════════════════════════════════════════════════════════════════


BACKGROUND: set[asyncio.Task] = set()
"""Hedge losers left to finish. Cancelling one mid-response makes httpx close its
connection, which drains the warm pool over a listening period; letting it finish
returns the connection to the pool."""


def finish_in_background(awaitable: Awaitable) -> None:
    """Run to completion off the critical path, keeping a reference so it is not collected."""
    task = asyncio.ensure_future(awaitable)
    BACKGROUND.add(task)
    task.add_done_callback(BACKGROUND.discard)


async def drain_stream(head: asyncio.Task, stream: AsyncIterator) -> None:
    """Consume a losing hedged stream to its end, so its connection is reused."""
    with contextlib.suppress(Exception):
        await head
        async for _ in stream:
            pass


async def first_result(make: Callable[[], Awaitable], fanout: int):
    """Await `fanout` identical calls and return the first success; losers finish in the background."""
    if fanout <= 1:
        return await make()
    pending = {asyncio.ensure_future(make()) for _ in range(fanout)}
    error: BaseException | None = None
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is None:
                    return task.result()
                error = task.exception()
        raise error  # type: ignore[misc]
    finally:
        for task in pending:
            finish_in_background(task)


async def first_stream(make: Callable[[], AsyncIterator], fanout: int) -> AsyncIterator:
    """Start `fanout` identical streams; follow whichever yields first, drain the rest."""
    if fanout <= 1:
        async with contextlib.aclosing(make()) as stream:
            async for item in stream:
                yield item
        return
    streams = [make() for _ in range(fanout)]
    heads = {asyncio.ensure_future(anext(stream)): stream for stream in streams}
    winner = None
    try:
        while heads and winner is None:
            done, _ = await asyncio.wait(heads, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                stream = heads.pop(task)
                if task.exception() is None:
                    winner, head = stream, task.result()
                    break
                with contextlib.suppress(Exception):
                    await stream.aclose()
        if winner is None:
            raise RuntimeError("every hedged stream failed")
    finally:
        for task, stream in heads.items():
            finish_in_background(drain_stream(task, stream))
    try:
        yield head
        async for item in winner:
            yield item
    finally:
        with contextlib.suppress(Exception):
            await winner.aclose()


# ══════════════════════════════════════════════════════════════════════════════════
#  Service calls (on long-lived, pre-warmed connections)
# ══════════════════════════════════════════════════════════════════════════════════


async def decide_once(openai: httpx.AsyncClient, transcript: str) -> tuple[str, float]:
    """One Decisions API choice call; returns (choice, P(interrupt))."""
    COST["decision_calls"] += 1
    IN_FLIGHT["decisions"] += 1
    try:
        return await _decide_request(openai, transcript)
    finally:
        IN_FLIGHT["decisions"] -= 1


async def _decide_request(openai: httpx.AsyncClient, transcript: str) -> tuple[str, float]:
    """The Decisions API request itself."""
    response = await openai.post(
        "/decisions",
        json={
            "model": DECISION_MODEL,
            "input": f"<user_goal>\n{GOAL_BLOCK}\n</user_goal>\n\nAgent (speaking now): {transcript}",
            "questions": [
                {"type": "choice", "name": "action", "instructions": DECISION_INSTRUCTIONS, "choices": DECISION_CHOICES}
            ],
        },
    )
    response.raise_for_status()
    answer = response.json()["answers"][0]
    if answer.get("type") == "refusal":
        raise RuntimeError(f"Decisions API refused: {answer}")
    probabilities = {p["value"]: float(p["probability"]) for p in answer.get("probabilities") or []}
    return str(answer.get("choice", "")), probabilities.get("interrupt", 0.0)


async def decide(openai: httpx.AsyncClient, transcript: str, *, fanout: int = 1) -> tuple[str, float]:
    """Decision, optionally hedged."""
    return await first_result(lambda: decide_once(openai, transcript), fanout)


async def stream_response_once(openai: httpx.AsyncClient, transcript: str, strategy: Strategy) -> AsyncIterator[str]:
    """Yield content deltas of the caller's interruption sentence."""
    COST["llm_calls"] += 1
    body = {
        "model": strategy.response_model,
        "messages": [{"role": "user", "content": RESPONSE_PROMPT.format(goal=GOAL_BLOCK, transcript=transcript)}],
        "max_completion_tokens": 60,
        "service_tier": strategy.service_tier,
        "stream": True,
    }
    if strategy.reasoning_effort is not None:
        body["reasoning_effort"] = strategy.reasoning_effort
    IN_FLIGHT["llm_streams"] += 1
    try:
        async with openai.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line[6:]).get("choices") or []
                text = (choices[0].get("delta") or {}).get("content") if choices else None
                if text:
                    yield text
    finally:
        IN_FLIGHT["llm_streams"] -= 1


def tts_body(text: str, *, voice_id: str) -> dict:
    """Cartesia request body for PCM16 mono at the caller sample rate (assistant turn only)."""
    return {
        "model_id": CARTESIA_MODEL,
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "language": LANGUAGE,
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": CALLER_SAMPLE_RATE},
    }


async def stream_tts_once(cartesia: CartesiaStreamingTTS, text: str) -> AsyncIterator[bytes]:
    """Yield sample-aligned PCM16 chunks of caller speech from the simulator's TTS."""
    COST["tts_calls"] += 1
    COST["tts_chars"] += len(text)
    carry = b""
    async for chunk in cartesia.stream(text, voice_id=VOICE_ID):
        # Chunk boundaries need not fall on samples; an odd split would shear PCM16.
        data = carry + chunk
        cut = len(data) - len(data) % 2
        carry = data[cut:]
        if cut:
            yield data[:cut]


async def synthesize(cartesia: CartesiaStreamingTTS, text: str) -> bytes:
    """Render caller speech in full."""
    return b"".join([chunk async for chunk in stream_tts_once(cartesia, text)])


async def render_assistant_turn(cartesia: httpx.AsyncClient) -> tuple[bytes, list[float]]:
    """Render the assistant turn with word timestamps; returns (pcm, word end times in ms)."""
    body = tts_body(ASSISTANT_TURN_TEXT, voice_id=ASSISTANT_VOICE_ID) | {"add_timestamps": True}
    pcm = bytearray()
    word_ends: list[float] = []
    async with cartesia.stream("POST", CARTESIA_SSE_URL, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event.get("type") == "chunk" and event.get("data"):
                pcm.extend(base64.b64decode(event["data"]))
            elif event.get("type") == "timestamps":
                word_ends.extend(end * 1000.0 for end in event["word_timestamps"]["end"])
            elif event.get("type") == "error":
                raise RuntimeError(f"Cartesia error: {event}")
    return bytes(pcm), word_ends


async def render_opener(cartesia: CartesiaStreamingTTS, phrase: str) -> Opener:
    """Pre-render an opener and trim its dead air."""
    raw = await synthesize(cartesia, phrase)
    return Opener(phrase=phrase, pcm=trim_silence(raw), raw_ms=pcm_duration_ms(len(raw)))


async def warm_up(
    openai: httpx.AsyncClient,
    decisions_client: httpx.AsyncClient,
    cartesia: CartesiaStreamingTTS,
    strategy: Strategy,
) -> None:
    """Open, concurrently, as many pooled connections as the strategy can have in flight at once.

    Decisions and replies share one pool. Warming fewer than the peak would make the
    decision open a fresh connection (a TLS handshake, ~100ms) inside the measurement
    whenever speculative streams already hold the warm ones.
    """
    warm = "Thanks for holding, I've pulled up your account now and"
    decisions = HEDGE_FANOUT if strategy.hedge_decision else 1
    reply_fanout = HEDGE_FANOUT if strategy.hedge_reply else 1
    speculative = strategy.start in ("rolling", "threshold", "trigger")
    streams = (ROLLING_MAX_IN_FLIGHT + 1 if speculative else 1) * reply_fanout
    probes = decisions  # a delta's decision can overlap the next one's
    tts = (ROLLING_MAX_IN_FLIGHT + 1 if speculative and strategy.speculate_tts else 1) * reply_fanout

    async def drain_response() -> None:
        async for _ in stream_response_once(openai, warm, strategy):
            pass

    # Concurrent, so each request needs its own connection and the pool reaches the peak.
    await asyncio.gather(
        *(decide_once(decisions_client, warm) for _ in range(decisions + probes)),
        *(drain_response() for _ in range(streams)),
    )
    await asyncio.gather(*(synthesize(cartesia, "Okay.") for _ in range(tts)))


class CallerEar:
    """The simulator's ScribeStreamingSTT, fed what the bridge delivers.

    The rendered turn is converted to 8kHz mu-law, the assistant's wire format, with resampler
    state carried across chunks; ScribeStreamingSTT.send() does the rest exactly as in a call.
    """

    def __init__(self) -> None:
        self.stt = ScribeStreamingSTT(STT_MODEL, language=LANGUAGE)
        self._ratecv_state = None

    async def start(self) -> None:
        await self.stt.start()

    async def feed(self, pcm: bytes) -> None:
        resampled, self._ratecv_state = audioop.ratecv(
            pcm, 2, 1, CALLER_SAMPLE_RATE, INPUT_SAMPLE_RATE, self._ratecv_state
        )
        await self.stt.send(audioop.lin2ulaw(resampled, 2))

    def heard_text(self) -> str:
        return self.stt.heard_text()

    async def stop(self) -> None:
        await self.stt.stop()


class ScribePcm16Ear(CallerEar):
    """Same SDK, model and callbacks as the simulator, but sent 16kHz PCM16 with no conversion."""

    async def start(self) -> None:
        stt = self.stt
        stt._connection = await ScribeRealtime(api_key=stt._api_key).connect(
            {
                "model_id": stt.model,
                "audio_format": AudioFormat.PCM_16000,
                "sample_rate": CALLER_SAMPLE_RATE,
                "commit_strategy": CommitStrategy.MANUAL,
                "language_code": stt.language,
            }
        )
        stt._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, stt._on_partial)
        stt._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, stt._on_committed)
        stt._connection.on(RealtimeEvents.ERROR, stt._on_error)

    async def feed(self, pcm: bytes) -> None:
        # Bypass send(): it decodes 8kHz mu-law, and this path is already 16kHz PCM16.
        await self.stt._connection.send({"audio_base_64": base64.b64encode(pcm).decode("ascii")})


EARS = {"production": CallerEar, "scribe_pcm16": ScribePcm16Ear}


async def open_stt(backend: str) -> CallerEar:
    """Start STT and stream silence until the session is live, so the clock starts warm."""
    stt = EARS[backend]()
    await stt.start()
    silence = b"\x00" * ms_to_bytes(FEED_CHUNK_MS)
    for _ in range(STT_WARMUP_MS // FEED_CHUNK_MS):
        await stt.feed(silence)
        await asyncio.sleep(FEED_CHUNK_MS / 1000)
    return stt


# ══════════════════════════════════════════════════════════════════════════════════
#  Measurement
# ══════════════════════════════════════════════════════════════════════════════════


def pool_size(client: httpx.AsyncClient) -> int:
    """Open connections in an httpx client's pool (private API: diagnostics only)."""
    try:
        return len(client._transport._pool.connections)
    except AttributeError:
        return -1


def active_strategies() -> list[str]:
    """STRATEGIES filtered by RUN_ONLY."""
    return [name for name in STRATEGIES if not RUN_ONLY or name in RUN_ONLY]


def strip_leading_filler(text: str) -> str:
    """Remove a repeated opener, or any leading interjection, from the content."""
    return LEADING_FILLER.sub("", text.strip()).strip()


async def run_text(reply: Reply, openai: httpx.AsyncClient, clock: Callable[[], float], strategy: Strategy) -> None:
    """Generate the reply's first sentence."""
    fanout = HEDGE_FANOUT if strategy.hedge_reply else 1
    text = ""
    # aclosing: breaking at the first sentence must release the HTTP stream now, not at GC.
    async with contextlib.aclosing(
        first_stream(lambda: stream_response_once(openai, reply.transcript, strategy), fanout)
    ) as llm:
        async for delta in llm:
            if not text:
                reply.gen_first_token_ms = clock() - reply.started_at_ms
            text += delta
            reply.partial = text
            reply.updated.set()
            if text.rstrip().endswith(SENTENCE_END):
                break
    reply.gen_first_sentence_ms = clock() - reply.started_at_ms
    reply.raw_text = text.strip()
    reply.text = strip_leading_filler(text)
    reply.updated.set()


async def run_audio(
    reply: Reply, cartesia: CartesiaStreamingTTS, clock: Callable[[], float], strategy: Strategy
) -> None:
    """Stream the reply text's audio into reply.chunks."""
    fanout = HEDGE_FANOUT if strategy.hedge_reply else 1
    tts_sent = clock()
    async for chunk in first_stream(lambda: stream_tts_once(cartesia, reply.text), fanout):
        arrived = clock()
        if not reply.chunks:
            reply.tts_ttfa_ms = arrived - tts_sent
        reply.chunks.append((arrived, chunk))


def first_clause(text: str) -> str:
    """The text up to the first clause break after CLAUSE_MIN_WORDS words, or '' if none yet."""
    for index, char in enumerate(text):
        if char in CLAUSE_BREAKS and len(text[:index].split()) >= CLAUSE_MIN_WORDS:
            return text[: index + 1]
    return ""


async def speak_by_clause(
    reply: Reply, cartesia: CartesiaStreamingTTS, clock: Callable[[], float], strategy: Strategy
) -> None:
    """Synthesize the first clause as soon as it exists, then the rest of the sentence.

    Segments are requested concurrently but kept in order, so playout is clause 1 then the
    remainder; a remainder that arrives late is an underrun the replay counts as gap.
    """
    fanout = HEDGE_FANOUT if strategy.hedge_reply else 1
    while True:
        head = reply.text or first_clause(strip_leading_filler(reply.partial))
        if head:
            break
        if reply.task and reply.task.done():
            reply.task.result()  # surface the generation error
            raise RuntimeError("reply finished without text")
        reply.updated.clear()
        await asyncio.wait(
            {asyncio.ensure_future(reply.updated.wait()), reply.task}, return_when=asyncio.FIRST_COMPLETED
        )

    segments: list[list[tuple[float, bytes]]] = []

    async def synth(text: str, first: bool) -> None:
        out: list[tuple[float, bytes]] = []
        segments.append(out)
        sent = clock()
        async for chunk in first_stream(lambda: stream_tts_once(cartesia, text), fanout):
            arrived = clock()
            if first and not out:
                reply.tts_ttfa_ms = arrived - sent
            out.append((arrived, chunk))

    head_task = asyncio.create_task(synth(head, True))
    if reply.task and not reply.task.done():
        await reply.task
    rest = reply.text[len(head) :].strip() if reply.text.startswith(head) else ""
    tail_task = asyncio.create_task(synth(rest, False)) if rest else None
    await head_task
    if tail_task:
        await tail_task
    reply.chunks = [chunk for segment in segments for chunk in segment]


async def run_reply(
    reply: Reply,
    openai: httpx.AsyncClient,
    cartesia: CartesiaStreamingTTS,
    clock: Callable[[], float],
    strategy: Strategy,
    *,
    with_audio: bool,
) -> None:
    """Text, then (optionally) audio."""
    await run_text(reply, openai, clock, strategy)
    if with_audio:
        await run_audio(reply, cartesia, clock, strategy)


async def run_iteration(
    sample: Sample,
    turn_pcm: bytes,
    word_ends: list[float],
    openai: httpx.AsyncClient,
    decisions_client: httpx.AsyncClient,
    cartesia: CartesiaStreamingTTS,
) -> list[tuple[float, bytes]]:
    """Stream the turn, interrupt on the trigger delta, return the chosen reply's chunks."""
    strategy = STRATEGIES[sample.strategy]
    decision_fanout = HEDGE_FANOUT if strategy.hedge_decision else 1
    stt = await open_stt(sample.stt_backend)  # setup: outside the clock
    replies: list[Reply] = []
    side_decisions: list[asyncio.Task] = []
    try:
        await warm_up(openai, decisions_client, cartesia, strategy)  # re-warm right before t0
        for key in COST:
            COST[key] = 0
        t0 = time.perf_counter()

        def clock() -> float:
            return (time.perf_counter() - t0) * 1000.0

        def start_reply(transcript: str, delta: int, *, speculative: bool) -> Reply | None:
            in_flight = [r for r in replies if r.task and not r.task.done()]
            if speculative and len(in_flight) >= ROLLING_MAX_IN_FLIGHT:
                if not strategy.keep_newest:
                    return None
                in_flight[0].task.cancel()  # make room for the freshest transcript
            reply = Reply(transcript=transcript, started_at_ms=clock(), delta=delta)
            with_audio = strategy.speculate_tts or not speculative
            reply.task = asyncio.create_task(run_reply(reply, openai, cartesia, clock, strategy, with_audio=with_audio))
            replies.append(reply)
            sample.replies_started += 1
            if speculative and not sample.first_speculation_delta:
                sample.first_speculation_delta = delta
            return reply

        async def listen_decision(transcript: str, delta: int, paired: Reply | None) -> None:
            """A pre-trigger delta's decision: recorded, then treated as keep listening.

            Hedged like the committing decision, since live any delta could be the one.
            """
            choice, p = await decide(decisions_client, transcript, fanout=decision_fanout)
            sample.delta_p.append((delta, p))
            if choice == "interrupt":
                sample.early_interrupt_votes += 1
            if paired is not None and paired.task:
                paired.task.cancel()  # its decision said keep listening
            if strategy.start == "threshold" and p >= SPECULATE_P and decision is None:
                start_reply(transcript, delta, speculative=True)

        step = ms_to_bytes(FEED_CHUNK_MS)
        deltas, last_text = 0, ""
        decision: asyncio.Task | None = None
        for offset in range(0, len(turn_pcm), step):
            # Pace against the audio clock, so drift does not accumulate.
            delay = t0 + offset / BYTES_PER_SECOND - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            await stt.feed(turn_pcm[offset : offset + step])
            if decision is not None:
                if decision.done():
                    break
                continue
            text = stt.heard_text()
            if not text.strip() or text == last_text:
                continue
            deltas, last_text = deltas + 1, text
            if deltas >= TRIGGER_ON_DELTA:
                sample.trigger_at_ms = clock()
                sample.transcript = text
                heard_words = min(len(text.split()), len(word_ends))
                sample.last_word_end_at_ms = word_ends[heard_words - 1] if heard_words else 0.0
                # Decision goes on the wire first: it gates the opener, the reply does not.
                sample.in_flight_at_trigger = dict(IN_FLIGHT)
                sample.pool_before = pool_size(decisions_client)
                decision = asyncio.create_task(decide(decisions_client, text, fanout=decision_fanout))
                await asyncio.sleep(0)
                if strategy.start == "trigger" or (strategy.start == "threshold" and not replies):
                    start_reply(text, deltas, speculative=False)
                elif strategy.start == "rolling":
                    start_reply(text, deltas, speculative=True)
            else:
                paired = None
                if strategy.start == "rolling":
                    start_reply(text, deltas, speculative=True)
                elif strategy.start == "trigger":
                    paired = start_reply(text, deltas, speculative=False)
                side_decisions.append(asyncio.create_task(listen_decision(text, deltas, paired)))
        sample.deltas_listened = deltas
        if decision is None:
            sample.error = f"only {deltas} transcript deltas, never reached TRIGGER_ON_DELTA={TRIGGER_ON_DELTA}"
            return []

        sample.choice, sample.p_interrupt = await decision
        sample.pool_after = pool_size(decisions_client)
        sample.opener_at_ms = clock()
        sample.decision_ms = sample.opener_at_ms - sample.trigger_at_ms
        if sample.choice != "interrupt" and not FORCE_INTERRUPT:
            return []
        if not replies:
            start_reply(sample.transcript, deltas, speculative=False)  # after_decision: only now

        # Freshest reply that is furthest along; cancel the rest.
        live = [r for r in replies if not (r.task and r.task.cancelled())]
        fresh = [
            r for r in live if strategy.max_stale_deltas is not None and r.delta >= deltas - strategy.max_stale_deltas
        ]
        if fresh:
            # Within the freshness window, the one furthest along (most text written) wins.
            # Tie (e.g. neither has a token yet): the older one has the head start.
            chosen = max(fresh, key=lambda r: (r.progress, len(r.partial), -r.started_at_ms))
        else:
            chosen = max(live, key=lambda r: (r.progress, r.started_at_ms))
        for reply in replies:
            if reply is not chosen and reply.task:
                reply.task.cancel()
        if strategy.tts_on_clause and not chosen.chunks:
            await speak_by_clause(chosen, cartesia, clock, strategy)
        else:
            await chosen.task
            if not chosen.chunks:
                # Text-only speculation: synthesize the chosen text now.
                await run_audio(chosen, cartesia, clock, strategy)

        sample.reply_started_at_ms = chosen.started_at_ms
        # Staleness, on the audio clock: the last word fully spoken before the opener
        # started (latest) vs the last word of the transcript the reply was written from
        # (starting). Both are word indexes into the rendered turn's timestamps.
        turn_words = ASSISTANT_TURN_TEXT.split()
        latest = sum(1 for end in word_ends if end <= sample.opener_at_ms)
        starting = min(len(chosen.transcript.split()), len(word_ends))
        sample.latest_word = turn_words[latest - 1] if latest else ""
        sample.starting_word = turn_words[starting - 1] if starting else ""
        sample.stale_words = max(0, latest - starting)
        sample.missed_words = " ".join(turn_words[starting:latest])
        sample.reply_transcript = chosen.transcript
        source_words = min(len(chosen.transcript.split()), len(word_ends))
        sample.source_end_at_ms = word_ends[source_words - 1] if source_words else 0.0
        sample.gen_first_token_ms = chosen.gen_first_token_ms
        sample.gen_first_sentence_ms = chosen.gen_first_sentence_ms
        sample.tts_ttfa_ms = chosen.tts_ttfa_ms
        sample.content_first_audio_at_ms = chosen.chunks[0][0] if chosen.chunks else 0.0
        sample.raw_content, sample.content = chosen.raw_text, chosen.text
        sample.ok = bool(chosen.chunks)
        if not sample.ok:
            sample.error = "chosen reply produced no audio"
        return chosen.chunks
    except Exception as exc:  # noqa: BLE001 - a failed iteration is a data point, not a crash
        sample.error = f"{type(exc).__name__}: {exc}"
        return []
    finally:
        pending = [r.task for r in replies if r.task] + side_decisions
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # Outside the clock: let hedge losers finish so their cost is counted and their
        # connections return to the pool for the next iteration.
        await asyncio.gather(*list(BACKGROUND), return_exceptions=True)
        sample.cost = dict(COST)
        await stt.stop()  # teardown: outside the clock


# ══════════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════════


def row(label: str, values: list[float]) -> str:
    """p50/min/max line. No p90: at a handful of samples it is just the max."""
    values = [v for v in values if v == v]  # drop NaN
    if not values:
        return f"    {label:<34} no data"
    return (
        f"    {label:<34} p50 {statistics.median(values):>6.0f}ms   min {min(values):>6.0f}ms   "
        f"max {max(values):>6.0f}ms   (n={len(values)})"
    )


def summarize_stt(samples: list[Sample]) -> None:
    """The A/B that matters: STT lag per backend, and where each one put the trigger."""
    print("\n" + "=" * 100 + "\n  STT BACKEND COMPARISON (same session, interleaved)\n" + "=" * 100)
    for backend in STT_BACKENDS:
        good = [s for s in samples if s.stt_backend == backend and s.ok]
        print(row(f"{backend} STT lag", [s.stt_lag_ms for s in good]))
        print(row(f"{backend} reaction", [s.reaction_ms for s in good]))
        triggers = sorted({s.transcript[-40:] for s in good})
        print(f"    {backend} trigger transcript(s): {triggers}")
        failed = [s for s in samples if s.stt_backend == backend and not s.ok]
        if failed:
            print(f"    {backend} failed {len(failed)}: {failed[0].error[:100]}")
    print("    NOTE: lag is timed from the end of the last transcribed word; compare backends only")
    print("          when they triggered on the same transcript (see above).")


def summarize(samples: list[Sample], openers: list[Opener]) -> None:
    """Per strategy: reaction, reply pipeline, then the gap against every opener."""
    print("\n" + "=" * 100)
    print(f"  decision {DECISION_MODEL}   speculate P >= {SPECULATE_P}   max in flight {ROLLING_MAX_IN_FLIGHT}")
    print(
        f"  stt {STT_MODEL}   tts {CARTESIA_MODEL} @ {CALLER_SPEED}x   hedge fanout {HEDGE_FANOUT}   target gap <= {GAP_TARGET_MS:.0f}ms"
    )
    print("  openers: " + ", ".join(f"{o.phrase!r} {o.ms:.0f}ms" for o in openers))
    for strategy in active_strategies():
        config = STRATEGIES[strategy]
        runs = [s for s in samples if s.strategy == strategy]
        good = [s for s in runs if s.ok]
        print("\n" + "=" * 100)
        print(f"  {strategy.upper()}   {len(good)} ok / {len(runs) - len(good)} failed")
        print(f"  {config}")
        print("=" * 100)
        print(row("STT lag (word end -> transcript)", [s.stt_lag_ms for s in good]))
        print(row("decision", [s.decision_ms for s in good]))
        print(f"    per-run decision ms: {[round(s.decision_ms) for s in good]}")
        print(f"    in flight at trigger: {[s.in_flight_at_trigger for s in good]}")
        print(f"    decision pool before->after trigger decision: {[(s.pool_before, s.pool_after) for s in good]}")
        print(row("REACTION (word end -> opener out)", [s.reaction_ms for s in good]))
        print(row("gen first sentence (from reply start)", [s.gen_first_sentence_ms for s in good]))
        print(row("TTS first audio", [s.tts_ttfa_ms for s in good]))
        print(row("content ready (from opener start)", [s.content_ready_ms for s in good]))
        if config.start in ("rolling", "threshold"):
            print(
                f"    stale words (latest word before opener - reply's starting word): {[s.stale_words for s in good]}"
            )
            print(
                row("STALE ms (opener start - starting word end)", [s.opener_at_ms - s.source_end_at_ms for s in good])
            )
            for sample in good:
                print(
                    f"    #{sample.index} starting {sample.starting_word!r} -> latest {sample.latest_word!r}"
                    f"   missed: {sample.missed_words!r}"
                )
            print(row("SOURCE AGE when reply audible", [s.source_age_ms for s in good]))
            print(f"    first speculation at delta: {[s.first_speculation_delta for s in good]}")
        if good:
            print(
                f"    P(interrupt) by delta (#0): {[(d, round(p, 2)) for d, p in sorted(good[0].delta_p)]}"
                f"   early interrupt votes: {[s.early_interrupt_votes for s in good]}"
            )
        costs = [s.cost for s in runs if s.cost]
        if costs:
            mean = {k: statistics.mean(c[k] for c in costs) for k in costs[0]}
            print(
                f"    COST to listen {statistics.mean(s.deltas_listened for s in runs):.0f} deltas, then interrupt (mean): decisions {mean['decision_calls']:.1f}  "
                f"LLM calls {mean['llm_calls']:.1f}  TTS calls {mean['tts_calls']:.1f}  "
                f"TTS chars {mean['tts_chars']:.0f}"
            )
        for opener in openers:
            gaps = [s.gaps_ms[opener.phrase] for s in good]
            seamless = sum(1 for g in gaps if g <= GAP_TARGET_MS)
            print(row(f"GAP {opener.phrase!r} ({seamless}/{len(gaps)} seamless)", gaps))
        for sample in runs:
            if not sample.ok:
                print(f"    #{sample.index} failed: {sample.error[:120]}")

    good = [s for s in samples if s.ok]
    if not good:
        return
    print("\n  READ")
    for strategy in active_strategies():
        runs = [s for s in good if s.strategy == strategy]
        best = [o.phrase for o in openers if runs and all(s.gaps_ms[o.phrase] <= GAP_TARGET_MS for s in runs)]
        print(f"    {strategy:<20} seamless in every run with: {best or 'none'}")
    reaction = statistics.median(s.reaction_ms for s in good)
    print(f"    Reaction p50 {reaction:.0f}ms overall vs {BUDGET_MS:.0f}ms human-reaction reference.")
    stripped = sum(1 for s in good if s.raw_content != s.content)
    print(f"    Content opened with a filler/repeat in {stripped}/{len(good)} runs (stripped before TTS).")
    print(f"    Decisions: P(interrupt) p50 {statistics.median(s.p_interrupt for s in good):.2f}")
    print(f"    Transcript at trigger: {good[0].transcript!r}")


# ══════════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    """Warm everything, run every strategy, save recordings, print the summary."""
    load_dotenv(override=True)
    openai_key = os.environ.get("OPENAI_API_KEY")
    cartesia_key = os.environ.get("CARTESIA_API_KEY")
    if not openai_key or not cartesia_key:
        raise SystemExit("OPENAI_API_KEY and CARTESIA_API_KEY must both be set.")
    out_dir = Path(OUTPUT_DIR) / "subset" if RUN_ONLY else Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with (
        httpx.AsyncClient(
            base_url=OPENAI_URL,
            headers={"Authorization": f"Bearer {openai_key}"},
            timeout=30.0,
            limits=KEEPALIVE,
        ) as openai,
        httpx.AsyncClient(
            base_url=OPENAI_URL,
            headers={"Authorization": f"Bearer {openai_key}"},
            timeout=30.0,
            limits=KEEPALIVE,
        ) as decisions_client,
        httpx.AsyncClient(
            headers={"X-API-Key": cartesia_key, "Cartesia-Version": CARTESIA_VERSION},
            timeout=30.0,
        ) as assistant_tts,
    ):
        cartesia = CartesiaStreamingTTS(CARTESIA_MODEL, language=LANGUAGE)
        print("Rendering the assistant turn and openers...")
        turn_pcm, word_ends = await render_assistant_turn(assistant_tts)
        openers = [await render_opener(cartesia, phrase) for phrase in OPENERS]
        print(f"  turn {pcm_duration_ms(len(turn_pcm)) / 1000:.1f}s, {len(word_ends)} words timestamped")
        for opener in openers:
            print(f"  opener {opener.phrase!r}: {opener.ms:.0f}ms (rendered {opener.raw_ms:.0f}ms)")

        samples: list[Sample] = []
        for strategy in active_strategies():
            for index in range(ITERATIONS_PER_STRATEGY):
                for backend in STT_BACKENDS:
                    sample = Sample(strategy=strategy, index=index, stt_backend=backend)
                    chunks = await run_iteration(sample, turn_pcm, word_ends, openai, decisions_client, cartesia)
                    samples.append(sample)
                    if sample.ok:
                        for opener_index, opener in enumerate(openers):
                            track, gap = replay_playout(sample.opener_at_ms, opener, chunks)
                            sample.gaps_ms[opener.phrase] = gap
                            if opener_index == 0:
                                sample.source_age_ms = sample.opener_at_ms + opener.ms + gap - sample.source_end_at_ms
                            wav = out_dir / backend / strategy / f"opener{opener_index}_run{index}.wav"
                            write_stereo(wav, turn_pcm, track)
                    gaps = " ".join(f"{g:4.0f}" for g in sample.gaps_ms.values())
                    print(
                        f"  [{strategy:<19} {backend:<13} #{index}] stt {sample.stt_lag_ms:4.0f}ms  "
                        f"reaction {sample.reaction_ms:5.0f}ms  gaps [{gaps}]ms  {sample.error or sample.content!r}"
                    )
        await cartesia.close()

    for backend in STT_BACKENDS:
        print("\n" + "#" * 100 + f"\n  STT BACKEND: {backend}\n" + "#" * 100)
        summarize([s for s in samples if s.stt_backend == backend], openers)
    summarize_stt(samples)
    payload = {
        "config": {
            "strategies": {name: asdict(config) for name, config in STRATEGIES.items()},
            "openers": [{"phrase": o.phrase, "ms": o.ms, "raw_ms": o.raw_ms} for o in openers],
            "stt_model": STT_MODEL,
            "stt_backends": STT_BACKENDS,
            "tts_model": CARTESIA_MODEL,
            "tts_speed": CALLER_SPEED,
            "hedge_fanout": HEDGE_FANOUT,
            "rolling_max_in_flight": ROLLING_MAX_IN_FLIGHT,
            "decision_model": DECISION_MODEL,
            "speculate_p": SPECULATE_P,
            "trigger_on_delta": TRIGGER_ON_DELTA,
        },
        "samples": [
            asdict(s)
            | {"reaction_ms": s.reaction_ms, "stt_lag_ms": s.stt_lag_ms, "content_ready_ms": s.content_ready_ms}
            for s in samples
        ],
    }
    (out_dir / "samples.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nRecordings <backend>/<strategy>/opener<i>_run<k>.wav (L=assistant at {ASSISTANT_GAIN:.0%}, R=caller) -> {out_dir}/"
    )


if __name__ == "__main__":
    asyncio.run(main())
