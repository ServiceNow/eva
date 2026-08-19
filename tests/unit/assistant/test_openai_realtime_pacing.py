import asyncio
import time

import pytest

from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, message: str) -> None:
        self.sent.append(message)


def _server(paced: bool, tmp_path) -> OpenAIRealtimeAssistantServer:
    from eva.models.agents import AgentConfig
    from eva.models.config import ModelConfig

    db_path = tmp_path / "db.json"
    db_path.write_text("{}")

    return OpenAIRealtimeAssistantServer(
        current_date_time="2026-01-01T00:00:00",
        pipeline_config=ModelConfig(s2s="gpt-realtime", s2s_params={"model": "gpt-realtime", "api_key": "k"}),
        agent=AgentConfig(
            id="agent_itsm",
            name="agent_itsm",
            role="r",
            description="d",
            instructions="i",
            tool_module_path="eva.assistant.tools.itsm_tools",
        ),
        agent_config_path="configs/agents/itsm_agent.yaml",
        scenario_db_path=str(db_path),
        output_dir=tmp_path,
        port=9999,
        conversation_id="c1",
        paced_output=paced,
    )


@pytest.mark.asyncio
async def test_unpaced_output_drains_without_sleeping(tmp_path):
    server = _server(paced=False, tmp_path=tmp_path)
    server._running = True
    ws = FakeWS()
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    for _ in range(20):
        queue.put_nowait(b"\xff" * 160)

    started = time.monotonic()
    task = asyncio.create_task(server._pace_audio_output(ws, queue))
    await asyncio.sleep(0.05)
    task.cancel()

    # 20 paced chunks would take ~400ms; unpaced must finish inside the 50ms window.
    assert len(ws.sent) == 20
    assert time.monotonic() - started < 0.2


@pytest.mark.asyncio
async def test_paced_output_still_holds_real_time_cadence(tmp_path):
    server = _server(paced=True, tmp_path=tmp_path)
    server._running = True
    ws = FakeWS()
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    for _ in range(20):
        queue.put_nowait(b"\xff" * 160)

    task = asyncio.create_task(server._pace_audio_output(ws, queue))
    await asyncio.sleep(0.05)
    task.cancel()

    # ~20ms per chunk means only a handful land in a 50ms window.
    assert len(ws.sent) < 10


def test_user_recently_active_counts_audio_deltas_not_wall_time(tmp_path):
    server = _server(paced=False, tmp_path=tmp_path)

    server.note_user_audio()
    assert server.user_recently_active() is True

    # A long stall must not make a just-spoken user look idle.
    for _ in range(server.USER_ACTIVE_GUARD_DELTAS):
        server.note_assistant_delta()
    assert server.user_recently_active() is False
