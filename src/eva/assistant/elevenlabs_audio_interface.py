# ---------------------------------------------------------------------------
# Twilio <-> ElevenLabs audio bridge
# ---------------------------------------------------------------------------


import asyncio

from elevenlabs.conversational_ai.conversation import AsyncAudioInterface

# ElevenLabs recommends 4000 samples (250ms) per input_callback call.
# At 16 kHz PCM16 that is 4000 * 2 = 8000 bytes.
INPUT_CHUNK_BYTES = 8000
INPUT_CHUNK_DURATION = 0.25  # seconds

# µ-law silence byte (encodes to ~0 amplitude).  Sent as padding so
# ElevenLabs' VAD always receives audio and can detect end-of-speech.
MULAW_SILENCE = b"\xff"


class TwilioAudioBridge(AsyncAudioInterface):
    """Bridges Twilio WebSocket audio to an ElevenLabs AsyncConversation.

    * Twilio sends 8 kHz mulaw which the session handler converts to 16 kHz
      PCM and pushes via :meth:`feed_user_audio`.  A background task drains
      that queue, buffers it into 250 ms chunks, and forwards them to
      ElevenLabs through ``input_callback``.
    * ElevenLabs delivers 16 kHz PCM assistant audio via :meth:`output`.  The
      session handler pulls it from :meth:`get_output_audio`, converts to
      mulaw, and sends it back over the Twilio WebSocket.
    """

    def __init__(self) -> None:
        self._input_callback = None
        self._input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._input_task: asyncio.Task | None = None

    # -- AsyncAudioInterface contract ---------------------------------------

    async def start(self, input_callback):
        self._input_callback = input_callback
        self._running = True
        self._input_task = asyncio.create_task(self._feed_input())

    async def stop(self):
        self._running = False
        if self._input_task:
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass

    async def output(self, audio: bytes):
        """Called by ElevenLabs with 16 kHz PCM16 assistant audio."""
        await self._output_queue.put(audio)

    async def interrupt(self):
        """Barge-in: discard queued assistant audio."""
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # -- Public helpers for the session handler ------------------------------

    async def feed_user_audio(self, audio: bytes) -> None:
        """Enqueue user audio (8 kHz mulaw) for delivery to ElevenLabs."""
        await self._input_queue.put(audio)

    async def get_output_audio(self, timeout: float = 1.0) -> bytes | None:
        """Dequeue next assistant audio chunk, or *None* on timeout."""
        try:
            return await asyncio.wait_for(self._output_queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    # -- Internal ------------------------------------------------------------

    async def _feed_input(self) -> None:
        """Buffer small Twilio chunks into 250 ms frames for ElevenLabs.

        The agent is configured to accept µ-law 8 kHz audio.  Twilio media
        messages are small (~160 bytes each), so we accumulate until we have
        a full 250 ms chunk or the interval elapses.

        Crucially, when no user audio arrives we must still send silence so
        that ElevenLabs' VAD can detect end-of-speech.  Without continuous
        input the VAD stalls and never triggers transcription.
        """
        # 8 kHz µ-law, 1 byte per sample → 2000 bytes per 250 ms chunk
        mulaw_chunk_bytes = int(8000 * INPUT_CHUNK_DURATION)

        buf = bytearray()
        while self._running:
            try:
                remaining = max(0.01, INPUT_CHUNK_DURATION - len(buf) / 8000)
                chunk = await asyncio.wait_for(self._input_queue.get(), timeout=remaining)
                buf.extend(chunk)
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            if len(buf) >= mulaw_chunk_bytes:
                # Send full chunks of real audio
                while len(buf) >= mulaw_chunk_bytes and self._input_callback:
                    await self._input_callback(bytes(buf[:mulaw_chunk_bytes]))
                    del buf[:mulaw_chunk_bytes]
            elif buf:
                # Partial audio on timeout — send as-is without silence
                # padding.  Mixing silence into speech chunks causes the
                # VAD to trigger end-of-speech mid-utterance.
                if self._input_callback:
                    await self._input_callback(bytes(buf))
                    buf.clear()
            elif self._input_callback:
                # No audio arrived this interval — send a pure silence
                # chunk so ElevenLabs' VAD can detect end-of-speech.
                await self._input_callback(MULAW_SILENCE * mulaw_chunk_bytes)
