import pytest

from eva.user_simulator.cascade.adapter.base import Adapter


def test_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Adapter()


async def test_concrete_adapter_satisfies_the_interface():
    from eva.user_simulator.cascade.tick_result import TickResult

    class StubAdapter(Adapter):
        async def start(self) -> None:
            pass

        async def run_tick(self, tick_number: int, outgoing_audio: bytes | None) -> TickResult:
            return TickResult(
                tick_number=tick_number,
                assistant_audio=b"\x00" * 4,
                assistant_audio_raw_bytes=0,
                wall_clock_ms=0,
            )

        async def stop(self) -> None:
            pass

    adapter = StubAdapter()
    await adapter.start()
    result = await adapter.run_tick(0, None)
    await adapter.stop()

    assert result.tick_number == 0
