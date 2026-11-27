"""Host-side bridge for one Guava SIP call.

Runs the SIP/RTP media relay (``relay.py``) in a Docker container and exchanges
audio with it over a Unix socket:

    bridge = GuavaSipBridge(sip_code, sip_host=...)
    await bridge.connect()                 # start relay container, dial, wait for media
    await bridge.send_audio(pcm8k_bytes)   # EVA -> Guava
    pcm = await bridge.recv_audio()        # Guava -> EVA
    await bridge.close()

Audio is 8 kHz PCM16 mono in both directions (the SIP and Twilio rate), so no
resampling happens here.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_RELAY_DIR = str(Path(__file__).resolve().parent)  # holds relay.py, mounted at /app


class GuavaSipBridge:
    """One live Guava SIP call, bridged to a Unix socket via the relay container."""

    def __init__(
        self,
        sip_code: str,
        *,
        sip_host: str,
        sip_port: int = 5060,
        connect_timeout: float = 40.0,
    ) -> None:
        self.sip_code = sip_code
        self.sip_host = sip_host
        self.sip_port = sip_port
        # Relay image ref, provided by Guava; set $GUAVA_RELAY_IMAGE.
        self.image = os.environ.get("GUAVA_RELAY_IMAGE", "")
        if not self.image:
            raise ValueError("GuavaSipBridge needs a relay image: set $GUAVA_RELAY_IMAGE.")
        self._connect_timeout = connect_timeout

        # Per-call unix socket (namespaced by code tail to avoid collisions).
        self._sock_dir = f"/tmp/guava_relay/{sip_code[-12:]}"
        self._sock_path = f"{self._sock_dir}/relay.sock"

        self._inbound_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._server: asyncio.AbstractServer | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._client_connected = asyncio.Event()
        self._proc: asyncio.subprocess.Process | None = None
        self._recv_task: asyncio.Task | None = None
        self._stdout_task: asyncio.Task | None = None
        self._connected = False

    async def connect(self) -> None:
        """Start the unix server, spawn the relay container, and dial the code.

        Returns once the SIP call is answered and the relay has connected back on
        the unix socket (i.e. media can flow). Raises on dial failure / timeout.
        """
        os.makedirs(self._sock_dir, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._sock_path)

        async def _on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self._reader = reader
            self._writer = writer
            self._client_connected.set()

        self._server = await asyncio.start_unix_server(_on_client, path=self._sock_path)
        os.chmod(self._sock_path, 0o777)

        self._proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{_RELAY_DIR}:/app",
            "-v",
            f"{self._sock_dir}:{self._sock_dir}",
            "--entrypoint",
            "python",
            self.image,
            "/app/relay.py",
            "--number",
            self.sip_code,
            "--sip-host",
            self.sip_host,
            "--sip-port",
            str(self.sip_port),
            "--sock",
            self._sock_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Wait for the relay to report the SIP call is up, then for it to connect
        # back on the unix socket. Both must happen within connect_timeout.
        try:
            await asyncio.wait_for(self._await_relay_connected(), timeout=self._connect_timeout)
        except TimeoutError as e:
            await self.close()
            raise RuntimeError(f"Guava SIP relay did not connect within {self._connect_timeout}s") from e

        self._stdout_task = asyncio.create_task(self._drain_stdout())
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._connected = True
        logger.info("Guava SIP call up (code=%s…, host=%s)", self.sip_code[:16], self.sip_host)

    async def _await_relay_connected(self) -> None:
        """Read relay stdout until RELAY_CONNECTED, then await the unix connection."""
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise RuntimeError("relay exited before connecting")
            text = line.decode(errors="replace").rstrip()
            if text.startswith("RELAY_CONNECTED"):
                logger.info("relay: %s", text)
                break
            if text.startswith("RELAY_ERROR"):
                raise RuntimeError(f"relay error: {text}")
            if text:
                logger.debug("relay: %s", text)
        await self._client_connected.wait()

    async def _drain_stdout(self) -> None:
        """Log any further relay stdout (RELAY_HANGUP, etc.)."""
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    logger.info("relay: %s", text)
        except asyncio.CancelledError:
            pass

    async def _recv_loop(self) -> None:
        """Relay unix socket -> inbound queue (8 kHz PCM16)."""
        assert self._reader is not None
        try:
            while True:
                data = await self._reader.read(65536)
                if not data:
                    break
                await self._inbound_q.put(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("Guava bridge recv loop ended: %s", e)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send 8 kHz PCM16 mono to the Guava agent (no-op if not connected)."""
        if not self._connected or self._writer is None:
            return
        try:
            self._writer.write(pcm_bytes)
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            self._connected = False

    async def recv_audio(self) -> bytes:
        """Await the next inbound 8 kHz PCM16 mono chunk from Guava."""
        return await self._inbound_q.get()

    async def close(self) -> None:
        """Hang up: close the socket, stop the relay container + unix server."""
        self._connected = False
        for task in (self._recv_task, self._stdout_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            if self._proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._sock_path)
