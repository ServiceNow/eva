# ---------------------------------------------------------------------------
# Twilio <-> ElevenLabs audio bridge
# ---------------------------------------------------------------------------


import asyncio

from elevenlabs.conversational_ai.conversation import AsyncAudioInterface

# ElevenLabs recommends 4000 samples (250ms) per input_callback call.
# At 16 kHz PCM16 that is 4000 * 2 = 8000 bytes.
INPUT_CHUNK_BYTES = 8000
INPUT_CHUNK_DURATION = 0.25  # seconds


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

        ElevenLabs expects 16 kHz PCM16 in ~4000-sample (8000-byte) chunks.
        Twilio media messages are ~640 bytes each after conversion, so we
        accumulate until we have a full chunk or the interval elapses.
        """
        buf = bytearray()
        while self._running:
            try:
                # Collect audio until we fill a chunk or time out
                remaining = max(0.01, INPUT_CHUNK_DURATION - len(buf) / (16000 * 2))
                chunk = await asyncio.wait_for(self._input_queue.get(), timeout=remaining)
                buf.extend(chunk)
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            # Send when we have enough data, or on timeout if there's anything
            if len(buf) >= INPUT_CHUNK_BYTES:
                while len(buf) >= INPUT_CHUNK_BYTES and self._input_callback:
                    await self._input_callback(bytes(buf[:INPUT_CHUNK_BYTES]))
                    del buf[:INPUT_CHUNK_BYTES]
            elif buf:
                # Partial buffer on timeout — send what we have so we don't
                # add latency waiting for the next Twilio packet
                if self._input_callback:
                    await self._input_callback(bytes(buf))
                    buf.clear()
