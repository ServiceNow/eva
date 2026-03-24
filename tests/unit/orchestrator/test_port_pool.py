"""Unit tests for orchestrator components."""

import asyncio

import pytest

from eva.orchestrator.port_pool import PortPool, PortPoolContextManager


class TestPortPool:
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test port pool initialization."""
        pool = PortPool(base_port=9000, pool_size=10)
        await pool.initialize()

        assert pool.available_count == 10
        assert pool.in_use_count == 0

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test acquiring and releasing ports."""
        pool = PortPool(base_port=9000, pool_size=5)
        await pool.initialize()

        # Acquire a port
        port = await pool.acquire()
        assert port >= 9000
        assert port < 9005
        assert pool.in_use_count == 1
        assert pool.available_count == 4
        assert pool.is_port_in_use(port)

        # Release the port
        await pool.release(port)
        assert pool.in_use_count == 0
        assert pool.available_count == 5
        assert not pool.is_port_in_use(port)

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        """Test acquiring multiple ports."""
        pool = PortPool(base_port=9000, pool_size=5)
        await pool.initialize()

        ports = []
        for _ in range(5):
            port = await pool.acquire()
            ports.append(port)

        assert pool.in_use_count == 5
        assert pool.available_count == 0

        # All ports should be unique
        assert len(set(ports)) == 5

        # All ports should be in the expected range
        for port in ports:
            assert 9000 <= port < 9005

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test that acquire times out when no ports available."""
        pool = PortPool(base_port=9000, pool_size=2)
        await pool.initialize()

        # Acquire all ports
        await pool.acquire()
        await pool.acquire()

        # Should timeout when trying to acquire another
        with pytest.raises(asyncio.TimeoutError):
            await pool.acquire(timeout=0.1)

    @pytest.mark.asyncio
    async def test_release_unused_port(self):
        """Test releasing a port that wasn't acquired."""
        pool = PortPool(base_port=9000, pool_size=5)
        await pool.initialize()

        # Should not raise, just log warning
        await pool.release(9999)
        assert pool.available_count == 5

    @pytest.mark.asyncio
    async def test_auto_initialize(self):
        """Test that acquire auto-initializes the pool."""
        pool = PortPool(base_port=9000, pool_size=5)

        # Pool should auto-initialize on first acquire
        port = await pool.acquire()
        assert port >= 9000

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test the PortPoolContextManager."""
        pool = PortPool(base_port=9000, pool_size=5)
        await pool.initialize()

        async with PortPoolContextManager(pool) as port:
            assert pool.in_use_count == 1
            assert 9000 <= port < 9005

        # Port should be released after context
        assert pool.in_use_count == 0
