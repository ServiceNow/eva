"""Ephemeral public URL via a Cloudflare quick tunnel — for one-shot media-streaming runs.

EVA is a one-shot CLI, so instead of asking the user to run a tunnel and paste a URL, the
media-streaming transport can bring up a Cloudflare quick tunnel itself for the duration of
the run: it publishes a public ``https://<random>.trycloudflare.com`` that forwards to EVA's
local media port, then tears it down on exit. Quick tunnels need **no Cloudflare account**.

Requires the ``cloudflared`` binary on PATH (https://github.com/cloudflare/cloudflared).
"""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class CloudflaredNotFound(RuntimeError):
    pass


async def _read_url(proc: asyncio.subprocess.Process, timeout: float) -> str:
    """Cloudflared prints the assigned URL to stderr shortly after start."""
    assert proc.stderr is not None
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            line = await asyncio.wait_for(proc.stderr.readline(), timeout=deadline - asyncio.get_event_loop().time())
        except TimeoutError:
            break
        if not line:
            break
        m = _URL_RE.search(line.decode("utf-8", "replace"))
        if m:
            return m.group(0)
    raise TimeoutError("cloudflared did not report a tunnel URL in time")


@asynccontextmanager
async def cloudflare_quick_tunnel(local_port: int, *, url_timeout: float = 30.0) -> AsyncIterator[str]:
    """Start a Cloudflare quick tunnel to ``http://localhost:{local_port}``; yield its public URL.

    The tunnel process is terminated when the context exits.
    """
    binary = shutil.which("cloudflared")
    if not binary:
        raise CloudflaredNotFound(
            "cloudflared not found on PATH. Install it "
            "(https://github.com/cloudflare/cloudflared) or set webhook_base_url explicitly."
        )
    proc = await asyncio.create_subprocess_exec(
        binary, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{local_port}",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        url = await _read_url(proc, url_timeout)
        logger.info("Cloudflare quick tunnel up: %s -> http://localhost:%s", url, local_port)
        yield url
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                proc.kill()
