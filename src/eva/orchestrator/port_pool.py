"""Port pool for managing WebSocket server ports."""

import asyncio

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class PortPool:
    """Manages a pool of ports for parallel conversations.

    Each conversation needs its own WebSocket server on a unique port.
    This pool pre-allocates a range of ports and provides thread-safe
    acquisition and release.
    """

    def __init__(self, base_port: int = 9000, pool_size: int = 150):
        """Initialize the port pool.

        Args:
            base_port: Starting port number for the pool
            pool_size: Number of ports in the pool
        """
        self.base_port = base_port
        self.pool_size = pool_size
        self._available: asyncio.Queue[int] = asyncio.Queue()
        self._in_use: set[int] = set()
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the pool with available ports."""
        async with self._lock:
            if self._initialized:
                return

            for i in range(self.pool_size):
                port = self.base_port + i
                await self._available.put(port)

            self._initialized = True
            logger.info(
                f"Port pool initialized with {self.pool_size} ports "
                f"(range: {self.base_port}-{self.base_port + self.pool_size - 1})"
            )

    async def acquire(self, timeout: float | None = None) -> int:
        """Acquire an available port from the pool.

        Args:
            timeout: Maximum time to wait for a port (None = wait forever)

        Returns:
            An available port number

        Raises:
            asyncio.TimeoutError: If timeout expires before a port is available
        """
        if not self._initialized:
            await self.initialize()

        try:
            if timeout is not None:
                port = await asyncio.wait_for(self._available.get(), timeout=timeout)
            else:
                port = await self._available.get()

            self._in_use.add(port)
            logger.debug(f"Acquired port {port} ({len(self._in_use)} in use)")
            return port

        except TimeoutError:
            logger.warning(f"Timeout waiting for available port (timeout={timeout}s)")
            raise

    async def release(self, port: int) -> None:
        """Release a port back to the pool.

        Args:
            port: The port number to release
        """
        if port not in self._in_use:
            logger.warning(f"Attempted to release port {port} that was not in use")
            return

        self._in_use.discard(port)
        await self._available.put(port)
        logger.debug(f"Released port {port} ({len(self._in_use)} still in use)")

    @property
    def available_count(self) -> int:
        """Number of currently available ports."""
        return self._available.qsize()

    @property
    def in_use_count(self) -> int:
        """Number of currently in-use ports."""
        return len(self._in_use)

    def is_port_in_use(self, port: int) -> bool:
        """Check if a specific port is currently in use."""
        return port in self._in_use


class PortPoolContextManager:
    """Context manager for acquiring and releasing ports."""

    def __init__(self, pool: PortPool, timeout: float | None = None):
        self.pool = pool
        self.timeout = timeout
        self.port: int | None = None

    async def __aenter__(self) -> int:
        self.port = await self.pool.acquire(timeout=self.timeout)
        return self.port

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.port is not None:
            await self.pool.release(self.port)
            self.port = None
