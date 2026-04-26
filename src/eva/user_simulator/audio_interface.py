"""Audio interface for connecting ElevenLabs to the assistant via WebSocket.

This implements the ElevenLabs AudioInterface to bridge between:
- ElevenLabs (user simulator) generating audio
- Assistant server receiving and responding with audio

Uses JSON + base64 μ-law encoding (Twilio-style protocol).
"""

import asyncio
import base64
import json
import time
from collections.abc import Callable

import websockets
from websockets.protocol import State as WebSocketState

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from elevenlabs.conversational_ai.conversation import AudioInterface

from eva.user_simulator.perturbation import AudioPerturbator
from eva.utils.logging import get_logger

logger = get_logger(__name__)


# Audio format constants
PCM_SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes per sample
ASSISTANT_SAMPLE_RATE = 8000  # Assistant uses 8kHz μ-law
ELEVENLABS_OUTPUT_RATE = 16000  # ElevenLabs outputs 16kHz PCM
ELEVENLABS_INPUT_FORMAT = "mulaw"  # ElevenLabs configured to accept μ-law 8kHz directly

# Chunk sizes for real-time streaming
SEND_CHUNK_DURATION_MS = 20  # Send 20ms chunks to simulate real-time
SEND_CHUNK_SIZE_PCM = int(ELEVENLABS_OUTPUT_RATE * SEND_CHUNK_DURATION_MS / 1000) * PCM_SAMPLE_WIDTH  # 640 bytes

# Timing constants for silence detection and polling
SILENCE_DETECTION_THRESHOLD_S = 0.2  # 200ms to detect assistant audio end
USER_END_DETECTION_DELAY_INTERVALS = 30  # 600ms (30 x 20ms) - longer to avoid splitting natural pauses
USER_CATCHUP_SILENCE_CHUNKS = 0  # Don't send catch-up silence for user - let VAD detect naturally
ASSISTANT_CATCHUP_SILENCE_CHUNKS = 10  # 200ms catch-up silence when assistant stops
FAST_POLL_TIMEOUT_S = 0.005  # 5ms - fast polling during active audio
NORMAL_POLL_TIMEOUT_S = 0.01  # 10ms - normal polling
IDLE_POLL_TIMEOUT_S = 0.1  # 100ms - can wait longer when idle

# Logging intervals (in chunks)
LOG_INTERVAL_SILENCE = 50  # Log every 50 silence chunks (~1s at 20ms)
LOG_INTERVAL_AUDIO_SEND = 200  # Log every 200 sent chunks
LOG_INTERVAL_AUDIO_RECV = 100  # Log every 100 received chunks
LOG_INTERVAL_INPUT_STREAM = 4  # Log every 4 input chunks (~1s at 250ms)


class BotToBotAudioInterface(AudioInterface):
    """Custom audio interface that connects ElevenLabs to the assistant server via WebSocket.

    Flow:
    - ElevenLabs generates audio (simulated user) → output() → send to assistant
    - Assistant responds with audio → receive via WebSocket → input_callback() → ElevenLabs hears it

    Sends audio in small 20ms chunks to ensure real-time streaming behavior
    and proper audio synchronization.
    """

    INPUT_FRAMES_PER_BUFFER = 4000  # 250ms @ 16kHz (same as DefaultAudioInterface)
    INPUT_CHUNK_DURATION = 0.25  # 250ms intervals for input callback

    def __init__(
        self,
        websocket_uri: str,
        conversation_id: str,
        record_callback: Callable[[str, bytes], None] | None = None,
        event_logger=None,
        conversation_done_callback: Callable[[str], None] | None = None,
        perturbator: AudioPerturbator | None = None,
    ):
        """Initialize the audio interface.

        Args:
            websocket_uri: The WebSocket URI of the assistant server
            conversation_id: Unique identifier for this conversation
            record_callback: Optional callback for recording audio (source, data)
            event_logger: Optional ElevenLabsEventLogger for logging audio timing
            conversation_done_callback: Optional callback for signaling conversation end
            perturbator: Optional perturbator to apply to user audio before sending
        """
        self.websocket_uri = websocket_uri
        self.conversation_id = conversation_id
        self.record_callback = record_callback
        self.event_logger = event_logger
        self.conversation_done_callback = conversation_done_callback
        self._perturbator = perturbator

        self.websocket = None
        self.running = False
        self.receive_task = None
        self.send_task = None
        self.input_stream_task = None

        self.input_callback = None  # Callback for assistant audio for elevenlabs to hear
        self.send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.audio_buffer: asyncio.Queue[bytes] = asyncio.Queue()

        # Track audio timing state
        self._user_audio_active = False  # elevenlabs_user speaking
        self._assistant_audio_active = False  # framework_agent speaking
        self._user_audio_ended_time = None  # Track when user audio ended for silence sending
        self._assistant_audio_ended_time = None  # Track when assistant audio ended for silence sending

        # Shutdown state
        self._stopping = False
        self._send_errors_logged = 0

        # Latency tracking
        self._latency_measurements: list[float] = []

    async def start_async(self) -> None:
        """Async initialization - connect to assistant WebSocket."""
        self.running = True

        logger.info(f"Connecting to assistant WebSocket: {self.websocket_uri}")
        self.websocket = await websockets.connect(self.websocket_uri)

        # Send connection message (JSON protocol)
        await self.websocket.send(
            json.dumps(
                {
                    "event": "connected",
                    "protocol": "voice-bench-v1",
                    "conversation_id": self.conversation_id,
                }
            )
        )

        # Send start message
        await self.websocket.send(
            json.dumps(
                {
                    "event": "start",
                    "conversation_id": self.conversation_id,
                }
            )
        )

        logger.info("Connected to assistant, starting audio tasks")

        # Start background tasks
        self.receive_task = asyncio.create_task(self._receive_from_assistant())
        self.send_task = asyncio.create_task(self._send_to_assistant())
        self.input_stream_task = asyncio.create_task(self._continuous_input_stream())

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Start the audio interface (called by ElevenLabs).

        Args:
            input_callback: Callback that we call with audio from assistant for ElevenLabs to hear
        """
        self.input_callback = input_callback
        logger.info("ElevenLabs audio interface started with callback")

    def stop(self) -> None:
        """Stop the audio interface (called by ElevenLabs).

        Only signals conversation end here. The WebSocket is kept open so the
        assistant pipeline (Pipecat STT) can finish processing the last user
        utterance. The actual WebSocket close happens later in stop_async().
        """
        logger.info("ElevenLabs audio interface stop() called")
        self.running = False

        # Signal conversation end but do NOT close the WebSocket yet.
        # The assistant's STT needs the connection alive to finish processing
        # the last user utterance (~4-5 seconds after audio ends).
        if self.conversation_done_callback:
            logger.info("Signaling conversation end: session_ended")
            self.conversation_done_callback("session_ended")

    async def _close_websocket_on_stop(self) -> None:
        """Helper to close WebSocket when stop() is called synchronously."""
        if self.websocket:
            try:
                if self.websocket.state == WebSocketState.OPEN:
                    await self.websocket.send(
                        json.dumps(
                            {
                                "event": "stop",
                                "conversation_id": self.conversation_id,
                            }
                        )
                    )
                    await self.websocket.close()
                    logger.info("WebSocket closed from stop()")
                else:
                    logger.info("WebSocket already closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket from stop(): {e}")

    async def stop_async(self) -> None:
        """Async cleanup - close WebSocket and cancel tasks."""
        self._stopping = True
        self.running = False

        # Cancel tasks
        for task in [self.receive_task, self.send_task, self.input_stream_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Send stop message and close WebSocket
        if self.websocket:
            try:
                await self.websocket.send(
                    json.dumps(
                        {
                            "event": "stop",
                            "conversation_id": self.conversation_id,
                        }
                    )
                )
            except Exception:
                pass
            await self.websocket.close()
            self.websocket = None

        logger.info("Audio interface stopped")

    def output(self, audio: bytes) -> None:
        """Called by ElevenLabs when it generates audio (simulated user speaking).

        Args:
            audio: Raw audio bytes from ElevenLabs (16kHz 16-bit PCM mono)
        """
        if self.running:
            try:
                if self._perturbator is not None:
                    audio = self._perturbator.apply(audio)
                self.send_queue.put_nowait(audio)
                # Record user audio
                if self.record_callback:
                    self.record_callback("user", audio)
            except asyncio.QueueFull:
                logger.warning("Send queue full, dropping audio")

    def interrupt(self) -> None:
        """Called when ElevenLabs wants to interrupt playback.

        Since this represents a user (not a bot), we don't interrupt.
        The user should be able to keep talking even when the assistant responds.
        """
        # Don't clear the send queue - let the user keep talking
        pass

    @staticmethod
    def _convert_pcm_to_mulaw(pcm_data: bytes) -> bytes:
        """Convert PCM audio to mulaw format for sending to assistant.

        Args:
            pcm_data: 16-bit PCM audio data at 16kHz (from ElevenLabs output)

        Returns:
            mulaw encoded audio data at 8kHz (for assistant)
        """
        try:
            # Downsample from 16kHz to 8kHz
            pcm_8khz, _ = audioop.ratecv(
                pcm_data,
                PCM_SAMPLE_WIDTH,  # 16-bit PCM
                1,  # mono
                ELEVENLABS_OUTPUT_RATE,  # from 16kHz
                ASSISTANT_SAMPLE_RATE,  # to 8kHz
                None,
            )

            # Convert 16-bit PCM to μ-law
            mulaw_data = audioop.lin2ulaw(pcm_8khz, PCM_SAMPLE_WIDTH)
            return mulaw_data
        except Exception as e:
            logger.warning(f"Error converting PCM to mulaw: {e}")
            return b""

    def _should_send_assistant_silence(self) -> bool:
        """Return True if we should send assistant silence (user stopped, waiting for assistant).

        When both timestamps are set (interruption scenario), we send based on which
        party ended more recently - that determines what we're waiting for.
        """
        if self._user_audio_active or self._assistant_audio_active:
            return False
        if self._user_audio_ended_time is None:
            return False
        # If both ended, only send assistant silence if user ended more recently
        if self._assistant_audio_ended_time is not None:
            return self._user_audio_ended_time > self._assistant_audio_ended_time
        return True

    def _should_send_user_silence(self) -> bool:
        """Return True if we should send user silence (assistant stopped, waiting for user).

        When both timestamps are set (interruption scenario), we send based on which
        party ended more recently - that determines what we're waiting for.
        """
        if self._user_audio_active or self._assistant_audio_active:
            return False
        if self._assistant_audio_ended_time is None:
            return False
        # If both ended, only send user silence if assistant ended more recently
        if self._user_audio_ended_time is not None:
            return self._assistant_audio_ended_time > self._user_audio_ended_time
        return True

    def _should_send_ambient_noise(self) -> bool:
        """Return True when ambient noise should stream continuously (user is silent).

        Unlike _should_send_user_silence(), this is not gated on assistant state —
        ambient noise streams to the assistant at all times when the user is not
        speaking, including during assistant speech, to simulate an always-open mic.
        """
        return not self._user_audio_active and self._perturbator is not None and self._perturbator.has_ambient_noise

    async def _send_audio_frame(self, mulaw_data: bytes) -> bool:
        """Send an audio frame to the websocket.

        Args:
            mulaw_data: μ-law audio at ASSISTANT_SAMPLE_RATE (8kHz)

        Returns:
            True if sent successfully
        """
        if not self.websocket or not mulaw_data:
            return False
        # Don't attempt to send if websocket is closed or we're stopping
        if self._stopping or self.websocket.state != WebSocketState.OPEN:
            return False
        try:
            audio_base64 = base64.b64encode(mulaw_data).decode("utf-8")
            message = {"event": "media", "conversation_id": self.conversation_id, "media": {"payload": audio_base64}}
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Error sending audio frame: {e}")
            return False

    async def _send_silence_frame(self, chunk_size: int = SEND_CHUNK_SIZE_PCM) -> bool:
        """Send a silence tone frame to the websocket (for debugging silence periods).

        Args:
            chunk_size: Size of chunk in bytes (at 16kHz PCM)

        Returns:
            True if silence was sent, False otherwise
        """
        if self._perturbator is not None and self._perturbator.has_ambient_noise:
            silence_pcm = self._perturbator.get_ambient_chunk(chunk_size)
        else:
            silence_pcm = b"\x00" * chunk_size
        silence_mulaw = self._convert_pcm_to_mulaw(silence_pcm)

        if not silence_mulaw:
            return False
        return await self._send_audio_frame(silence_mulaw)

    async def _send_catchup_silence(self, source: str, num_chunks: int) -> None:
        """Send catch-up silence frames to cover detection delay.

        Sends chunks at real-time rate (20ms intervals) to maintain proper
        audio timing for STT/VAD systems.

        Args:
            source: "assistant" or "user" - who the silence represents
            num_chunks: Number of 20ms chunks to send
        """
        send_interval = SEND_CHUNK_DURATION_MS / 1000.0  # 20ms
        for i in range(num_chunks):
            await self._send_silence_frame(chunk_size=SEND_CHUNK_SIZE_PCM)
            # Space out chunks at real-time rate (skip delay on last chunk)
            if i < num_chunks - 1:
                await asyncio.sleep(send_interval)

    async def _on_user_audio_start(self) -> None:
        """Handle user audio starting."""
        self._user_audio_active = True
        self._user_audio_ended_time = None
        timestamp_ms = time.time()

        if self._assistant_audio_ended_time is not None:
            silence_duration = asyncio.get_event_loop().time() - self._assistant_audio_ended_time
            logger.info(f"🎤 User audio START - stopping user silence after {silence_duration:.2f}s")
            self._assistant_audio_ended_time = None
        if self.event_logger:
            self.event_logger.log_audio_start("elevenlabs_user", timestamp_ms)
        logger.info("🎤 User audio START")

        # Send user_speech_start event to assistant server with timestamp
        if self.websocket and self.websocket.state == WebSocketState.OPEN:
            try:
                await self.websocket.send(
                    json.dumps(
                        {
                            "event": "user_speech_start",
                            "conversation_id": self.conversation_id,
                            "timestamp_ms": str(int(round(timestamp_ms * 1000))),
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Error sending user_speech_start event: {e}")

    async def _on_user_audio_end(self, current_time: float) -> None:
        """Handle user audio ending."""
        self._user_audio_ended_time = current_time
        self._user_audio_active = False
        if self.event_logger:
            self.event_logger.log_audio_end("elevenlabs_user")
        logger.info("🎤 User audio END")

        # Send user_speech_stop event so assistant servers can compute model response latency.
        if self.websocket and self.websocket.state == WebSocketState.OPEN:
            try:
                await self.websocket.send(
                    json.dumps(
                        {
                            "event": "user_speech_stop",
                            "conversation_id": self.conversation_id,
                            "timestamp_ms": str(int(round(current_time * 1000))),
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Error sending user_speech_stop event: {e}")
        # Don't send catch-up silence for user audio end - let the continuous
        # silence sending in _send_to_assistant handle it naturally. This avoids
        # blocking and lets the VAD detect end-of-speech from actual silence.
        if USER_CATCHUP_SILENCE_CHUNKS > 0:
            await self._send_catchup_silence("assistant", USER_CATCHUP_SILENCE_CHUNKS)

    def _on_assistant_audio_start(self) -> None:
        """Handle assistant audio starting."""
        if self._user_audio_ended_time is not None:
            latency = asyncio.get_event_loop().time() - self._user_audio_ended_time
            self._latency_measurements.append(latency)
            logger.info(f"✅ Assistant responded after {latency:.2f}s - stopping assistant silence")
            self._user_audio_ended_time = None
        if self._assistant_audio_ended_time is not None:
            self._assistant_audio_ended_time = None
        self._assistant_audio_active = True
        if self.event_logger:
            self.event_logger.log_audio_start("framework_agent")
        logger.info("🔊 Assistant audio START")

    async def _on_assistant_audio_end(self) -> None:
        """Handle assistant audio ending (silence detected)."""
        self._assistant_audio_active = False
        self._assistant_audio_ended_time = asyncio.get_event_loop().time()
        if self.event_logger:
            self.event_logger.log_audio_end("framework_agent")
        logger.info("🔊 Assistant audio END (silence detected)")
        # Send catch-up silence to cover the detection delay for ElevenLabs
        if ASSISTANT_CATCHUP_SILENCE_CHUNKS > 0 and not self._should_send_ambient_noise():
            await self._send_catchup_silence("user", ASSISTANT_CATCHUP_SILENCE_CHUNKS)

    async def _continuous_input_stream(self) -> None:
        """Continuously call input_callback at regular intervals.

        This ensures ElevenLabs receives audio input at a steady rate, just like
        from a real microphone. When there's audio from the assistant, we send that.
        When there's no audio, we send silence.
        """
        # Calculate chunk size: μ-law 8kHz, 250ms chunks
        # 8000 samples/sec * 0.25s = 2000 samples (μ-law is 1 byte per sample)
        samples_per_chunk = int(ASSISTANT_SAMPLE_RATE * self.INPUT_CHUNK_DURATION)

        logger.info(
            f"Starting continuous input stream (chunk: {samples_per_chunk} bytes, interval: {self.INPUT_CHUNK_DURATION}s)"
        )

        # Track silence state for faster polling when no audio is available
        consecutive_empty_chunks = 0

        while self.running:
            start_time = asyncio.get_event_loop().time()

            # Collect audio from buffer
            audio_chunk = b""
            try:
                while len(audio_chunk) < samples_per_chunk:
                    remaining_time = self.INPUT_CHUNK_DURATION - (asyncio.get_event_loop().time() - start_time)
                    # Use shorter timeout when we've been getting empty buffers (silence mode)
                    if consecutive_empty_chunks > 0:
                        timeout = FAST_POLL_TIMEOUT_S
                    else:
                        timeout = max(NORMAL_POLL_TIMEOUT_S, remaining_time)

                    try:
                        chunk = await asyncio.wait_for(self.audio_buffer.get(), timeout=timeout)
                        audio_chunk += chunk
                        consecutive_empty_chunks = 0  # Reset on successful audio
                    except TimeoutError:
                        # In silence mode with short timeout, keep trying until chunk duration elapsed
                        if consecutive_empty_chunks > 0 and remaining_time > NORMAL_POLL_TIMEOUT_S:
                            continue
                        break
            except Exception as e:
                logger.error(f"Error getting audio from buffer: {e}")

            assistant_silence = False
            # Check if assistant audio has ended (silence detected after threshold)
            if self._assistant_audio_active and self._assistant_audio_ended_time:
                current_time = asyncio.get_event_loop().time()
                if current_time - self._assistant_audio_ended_time > SILENCE_DETECTION_THRESHOLD_S:
                    await self._on_assistant_audio_end()
            else:
                # Pad with silence if needed (μ-law silence = 0xFF)
                if len(audio_chunk) < samples_per_chunk:
                    padding_needed = samples_per_chunk - len(audio_chunk)
                    consecutive_empty_chunks += 1
                    audio_chunk += b"\xff" * padding_needed
                    if padding_needed == samples_per_chunk:
                        assistant_silence = True

            if self.input_callback:
                self.input_callback(audio_chunk)

            # Send assistant silence while waiting for assistant to respond
            # Send in 20ms chunks (same as user silence) for smoother timing
            if assistant_silence and self._should_send_assistant_silence() and not self._should_send_ambient_noise():
                # Calculate how many 20ms chunks fit in 250ms (round up to ensure we send enough)
                # 250ms / 20ms = 12.5, round up to 18 to avoid falling behind real-time
                chunks_to_send = 18
                for _ in range(chunks_to_send):
                    await self._send_silence_frame(chunk_size=SEND_CHUNK_SIZE_PCM)
                if consecutive_empty_chunks % LOG_INTERVAL_INPUT_STREAM == 1:
                    logger.debug("Sending silence assistant")

            # Maintain steady rate
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, self.INPUT_CHUNK_DURATION - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _receive_from_assistant(self) -> None:
        """Receive audio from the assistant server and buffer it."""
        audio_chunks_received = 0

        try:
            async for message in self.websocket:
                if not self.running:
                    break

                try:
                    data = json.loads(message)
                    event = data.get("event", data.get("type", ""))

                    # Handle media (audio) messages
                    if event == "media":
                        payload = data.get("media", {}).get("payload", "")
                        if payload:
                            mulaw_audio = base64.b64decode(payload)
                            if mulaw_audio:
                                # Mark start of assistant audio on first chunk
                                if not self._assistant_audio_active:
                                    self._on_assistant_audio_start()

                                audio_chunks_received += 1
                                self._assistant_audio_ended_time = asyncio.get_event_loop().time()
                                if audio_chunks_received % LOG_INTERVAL_AUDIO_RECV == 1:
                                    logger.debug(
                                        f"← Received audio chunk {audio_chunks_received} ({len(mulaw_audio)} bytes)"
                                    )

                                # Pass μ-law audio directly to ElevenLabs (configured for mulaw input)
                                await self.audio_buffer.put(mulaw_audio)

                    elif event == "transcript":
                        # Assistant sent a transcript (for logging)
                        text = data.get("text", "")
                        logger.debug(f"Assistant transcript: {text}")

                except json.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosedError as e:
            if e.code == 1012:  # Service restart (manual cancellation)
                logger.info("WebSocket closed due to service restart")
            elif self.running:
                logger.exception(f"Error receiving from assistant: {e}")
        except Exception as e:
            if self.running:
                logger.exception(f"Error receiving from assistant: {e}")
        finally:
            # Mark end of assistant audio if still active
            if self._assistant_audio_active and self.event_logger:
                self._assistant_audio_active = False
                self.event_logger.log_audio_end("framework_agent")
                logger.info("🔊 Assistant audio END (connection closed)")

            # Signal conversation end if disconnected while still running
            # This handles ElevenLabs disconnect due to timeout or network issues
            if self.running and self.conversation_done_callback:
                # WebSocket closed while conversation was active
                # This indicates ElevenLabs disconnect or network issue
                logger.warning("⚠️ WebSocket closed during active conversation - signaling disconnect")
                self.conversation_done_callback("elevenlabs_disconnect")

    async def _send_to_assistant(self) -> None:
        """Send audio from queue to assistant in small real-time chunks."""
        audio_chunks_sent = 0
        pcm_chunk_size = SEND_CHUNK_SIZE_PCM
        send_interval = SEND_CHUNK_DURATION_MS / 1000.0

        pending_audio = b""
        # Use absolute time targets to prevent drift from processing overhead
        stream_start_time: float | None = None  # Set when first audio chunk arrives
        silence_start_time: float | None = None  # Set when silence sending begins
        silence_chunks_sent = 0  # Separate counter for silence
        next_send_time = asyncio.get_event_loop().time()

        logger.info(
            f"Starting chunked audio sender (chunk={pcm_chunk_size} bytes, interval={send_interval * 1000:.0f}ms)"
        )

        while self.running:
            try:
                current_time = asyncio.get_event_loop().time()

                # Get more audio from queue
                # Calculate timeout based on time until next send to maintain accurate 20ms intervals
                time_until_next_send = max(0, next_send_time - current_time)
                if (
                    self._user_audio_ended_time is not None
                    or self._assistant_audio_ended_time is not None
                    or pending_audio
                ):
                    # Use remaining time until next send, with a small minimum to avoid busy-waiting
                    timeout = max(0.001, time_until_next_send)
                else:
                    timeout = IDLE_POLL_TIMEOUT_S
                try:
                    pcm_audio = await asyncio.wait_for(self.send_queue.get(), timeout=timeout)
                    pending_audio += pcm_audio
                    # Initialize/reset stream start time when audio arrives after idle/silence
                    if stream_start_time is None or not self._user_audio_active:
                        stream_start_time = asyncio.get_event_loop().time()
                        next_send_time = stream_start_time
                        audio_chunks_sent = 0  # Reset chunk counter for new stream
                        # Reset silence timing when transitioning to audio
                        silence_start_time = None
                        silence_chunks_sent = 0
                except TimeoutError:
                    pass

                # Refresh current_time after queue wait for accurate timing
                current_time = asyncio.get_event_loop().time()

                # Send chunks at regular intervals using absolute time targets
                if len(pending_audio) >= pcm_chunk_size and current_time >= next_send_time:
                    # Extract one chunk
                    chunk = pending_audio[:pcm_chunk_size]
                    pending_audio = pending_audio[pcm_chunk_size:]

                    if self.websocket:
                        # Mark start of user audio on first chunk
                        if not self._user_audio_active:
                            await self._on_user_audio_start()

                        # Convert to μ-law and send
                        mulaw_audio = self._convert_pcm_to_mulaw(chunk)
                        if mulaw_audio and await self._send_audio_frame(mulaw_audio):
                            audio_chunks_sent += 1
                            # Calculate next send time based on absolute target (prevents drift)
                            next_send_time = stream_start_time + (audio_chunks_sent * send_interval)
                            if audio_chunks_sent % LOG_INTERVAL_AUDIO_SEND == 0:
                                logger.debug(f"→ Sent audio chunk {audio_chunks_sent}")

                elif len(pending_audio) > 0 and len(pending_audio) < pcm_chunk_size:
                    # Partial chunk - wait for more or send after delay
                    time_since_last_send = current_time - next_send_time + send_interval
                    if time_since_last_send >= send_interval * USER_END_DETECTION_DELAY_INTERVALS:
                        if self.websocket and pending_audio:
                            # Pad partial chunk to full size with trailing silence
                            # This ensures consistent chunk sizes for STT processing
                            original_len = len(pending_audio)
                            # Align to sample boundary (2 bytes per sample)
                            aligned_len = (original_len // PCM_SAMPLE_WIDTH) * PCM_SAMPLE_WIDTH
                            padded_chunk = pending_audio[:aligned_len] + b"\x00" * (pcm_chunk_size - aligned_len)

                            mulaw_audio = self._convert_pcm_to_mulaw(padded_chunk)
                            if mulaw_audio and await self._send_audio_frame(mulaw_audio):
                                audio_chunks_sent += 1
                                logger.info(
                                    f"Sent padded chunk ({original_len} bytes padded to {pcm_chunk_size}) - end of utterance"
                                )
                                pending_audio = b""

                                # Mark end of user audio and start sending silence for VAD
                                if self._user_audio_active:
                                    await self._on_user_audio_end(current_time)
                                    # Reset stream timing for next audio stream
                                    stream_start_time = None
                                    next_send_time = current_time + send_interval

                # Send user silence/ambient noise while user is not speaking.
                # Ambient noise streams continuously (including during assistant speech).
                # Regular silence only sends when waiting for user to respond after assistant spoke.
                if self._should_send_ambient_noise() or self._should_send_user_silence():
                    # Initialize silence timing baseline when starting a NEW silence period
                    if silence_start_time is None:
                        silence_start_time = current_time
                        silence_chunks_sent = 0
                        next_send_time = silence_start_time

                    # Ambient noise: send regardless of assistant state (always-open mic)
                    # Regular silence: only send after assistant has spoken at least once
                    can_send = self._should_send_ambient_noise() or self._assistant_audio_ended_time is not None
                    if current_time >= next_send_time and can_send:
                        silence_pcm = b"\x00" * pcm_chunk_size
                        if await self._send_silence_frame():
                            silence_chunks_sent += 1
                            # Use absolute timing for silence (prevents drift)
                            next_send_time = silence_start_time + (silence_chunks_sent * send_interval)
                            # Record only after successful send to prevent double-recording on retry
                            if self.record_callback:
                                self.record_callback("assistant", silence_pcm)
                            if silence_chunks_sent % LOG_INTERVAL_SILENCE == 0:
                                actual_elapsed = current_time - silence_start_time
                                expected_elapsed = silence_chunks_sent * send_interval
                                logger.debug(
                                    f"Sending silence user: chunks={silence_chunks_sent}, actual={actual_elapsed:.3f}s, expected={expected_elapsed:.3f}s, ratio={actual_elapsed / expected_elapsed:.2f}x"
                                )
                else:
                    # Reset silence timing when user is speaking
                    if silence_start_time is not None:
                        silence_start_time = None
                        silence_chunks_sent = 0

                # Prevent busy-waiting - sleep until next scheduled send time
                if (
                    not pending_audio
                    and self._user_audio_ended_time is None
                    and self._assistant_audio_ended_time is None
                    and not self._should_send_ambient_noise()
                ):
                    await asyncio.sleep(NORMAL_POLL_TIMEOUT_S)
                else:
                    sleep_time = max(0, next_send_time - asyncio.get_event_loop().time())
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

            except Exception as e:
                if self.running:
                    logger.error(f"Error sending to assistant: {e}")

        # Send remaining audio on shutdown
        if pending_audio and self.websocket:
            try:
                mulaw_audio = self._convert_pcm_to_mulaw(pending_audio)
                if mulaw_audio and await self._send_audio_frame(mulaw_audio):
                    logger.info(f"Sent final {len(pending_audio)} bytes on shutdown")
            except Exception as e:
                logger.warning(f"Error sending final audio: {e}")

    def get_latencies(self) -> list[float]:
        """Return accumulated response latency measurements.

        Latency is measured as the time between when the user stops speaking
        and when the assistant's audio response begins.

        Returns:
            List of latency measurements in seconds.
        """
        return self._latency_measurements.copy()
