"""Tool stubs for EVA's ToolExecutor.

When framework=livekit, the real chariot voice agent (running in
docker) executes its own tools — these stubs only exist so EVA's
ToolExecutor can construct without ImportError and so the agent.yaml tool
declarations have somewhere to point.

EVA may call these stubs when running text-only / metric-only flows that
don't go through the bridge. In those cases the stubs return a clearly-
marked placeholder so it's obvious the tool didn't really execute.
"""

from __future__ import annotations

from typing import Any as _Any

_STUB_NOTE = (
    "stubbed-by-eva_chariot: real execution happens inside the chariot "
    "LiveKit agent, not through EVA's ToolExecutor"
)


def contact_details(
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    **_: _Any,
) -> dict[str, _Any]:
    return {
        "full_name": name,
        "email": email,
        "phone": phone,
        "_note": _STUB_NOTE,
    }


def end_call(**_: _Any) -> dict[str, _Any]:
    return {"status": "call_ended", "_note": _STUB_NOTE}


def save_voicemail(
    message: str,
    name: str | None = None,
    phone: str | None = None,
    email: str | None = None,
    **_: _Any,
) -> dict[str, _Any]:
    return {
        "saved": True,
        "voicemail": {"name": name, "phone": phone, "email": email, "message": message},
        "_note": _STUB_NOTE,
    }
