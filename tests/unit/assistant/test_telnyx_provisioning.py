"""Unit tests for the Telnyx media-streaming provisioner (no network)."""

from __future__ import annotations

import pytest

from eva.assistant.telnyx_provisioning.provision import (
    MANAGED_APP_NAME,
    ProvisionResult,
    TelnyxProvisioner,
    _NeedsCreate,
)


class _FakeResp:
    def __init__(self, data, status=200, text=""):
        self._data = data
        self.status_code = status
        self.text = text or str(data)

    def json(self):
        return {"data": self._data}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)


class _FakeClient:
    def __init__(self, numbers, apps, profiles):
        self._numbers = numbers
        self._apps = apps
        self._profiles = profiles
        self.posts: list[tuple[str, dict]] = []
        self.patches: list[tuple[str, dict]] = []

    def get(self, path, params=None):
        if path == "/call_control_applications":
            return _FakeResp(self._apps)
        if path == "/outbound_voice_profiles":
            return _FakeResp(self._profiles)
        if path == "/phone_numbers":
            return _FakeResp(self._numbers)
        return _FakeResp([])

    def post(self, path, json):
        self.posts.append((path, json))
        new = {"id": f"new-{len(self.posts)}", **json}
        return _FakeResp(new, status=201)

    def patch(self, path, json):
        self.patches.append((path, json))
        return _FakeResp({"id": path.rsplit("/", 1)[-1], **json})

    def close(self):
        pass


def _prov(numbers, apps, profiles):
    p = TelnyxProvisioner(api_key="test-key")
    p._client = _FakeClient(numbers, apps, profiles)
    return p


def test_requires_api_key():
    with pytest.raises(ValueError):
        TelnyxProvisioner(api_key="")


def test_inspect_reports_missing_connection():
    prov = _prov(
        numbers=[{"id": "n1", "phone_number": "+13125551234", "connection_id": None}],
        apps=[],
        profiles=[{"id": "prof-1", "name": "existing"}],
    )
    with pytest.raises(_NeedsCreate) as exc:
        prov.ensure(assistant_id="assistant-abc", public_url="https://x.example", create=False)
    assert exc.value.resource == "call_control_application"


def test_provision_creates_connection_and_reuses_profile_and_attaches_number():
    prov = _prov(
        numbers=[{"id": "n1", "phone_number": "+13125551234", "connection_id": None}],
        apps=[],
        profiles=[{"id": "prof-1", "name": "existing"}],
    )
    result = prov.ensure(assistant_id="assistant-abc", public_url="https://x.example/", create=True)
    assert isinstance(result, ProvisionResult)
    assert result.from_number == "+13125551234"
    assert result.to == "sip:anonymous@assistant-abc.sip.telnyx.com"
    assert result.webhook_base_url == "https://x.example"
    assert result.outbound_voice_profile_id == "prof-1"  # reused, not created
    # created the CC app and attached the number
    created_paths = [p for p, _ in prov._client.posts]
    assert "/call_control_applications" in created_paths
    assert prov._client.patches  # number attached


def test_prefers_unattached_number():
    prov = _prov(
        numbers=[
            {"id": "a", "phone_number": "+13120000001", "connection_id": "existing-conn"},
            {"id": "b", "phone_number": "+13120000002", "connection_id": None},
        ],
        apps=[
            {
                "id": "app-1",
                "application_name": MANAGED_APP_NAME,
                "webhook_event_url": "https://x.example/call-control-events",
            }
        ],
        profiles=[{"id": "prof-1", "name": "existing"}],
    )
    result = prov.ensure(assistant_id="assistant-abc", public_url="https://x.example", create=True)
    assert result.from_number == "+13120000002"  # the unattached one


def test_idempotent_when_everything_exists():
    prov = _prov(
        numbers=[{"id": "b", "phone_number": "+13120000002", "connection_id": "app-1"}],
        apps=[
            {
                "id": "app-1",
                "application_name": MANAGED_APP_NAME,
                "webhook_event_url": "https://x.example/call-control-events",
            }
        ],
        profiles=[{"id": "prof-1", "name": "eva-media-streaming"}],
    )
    result = prov.ensure(assistant_id="assistant-abc", public_url="https://x.example", create=True)
    assert result.created is None  # nothing created
    assert not prov._client.posts
    assert not prov._client.patches


class _FakeStderr:
    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self):
        return self._lines.pop(0) if self._lines else b""


class _FakeProc:
    def __init__(self, lines):
        self.stderr = _FakeStderr(lines)
        self.returncode = 0

    def terminate(self):
        pass

    def kill(self):
        pass

    async def wait(self):
        return 0


def test_tunnel_url_parse():
    import asyncio

    from eva.assistant.telnyx_provisioning.tunnel import _read_url

    proc = _FakeProc(
        [
            b"2026-... INF Thank you for trying Cloudflare Tunnel.\n",
            b"2026-... INF |  https://brave-red-fox-1234.trycloudflare.com  |\n",
        ]
    )
    url = asyncio.run(_read_url(proc, timeout=5))
    assert url == "https://brave-red-fox-1234.trycloudflare.com"


def test_tunnel_missing_binary(monkeypatch):
    import asyncio

    import eva.assistant.telnyx_provisioning.tunnel as tun

    monkeypatch.setattr(tun.shutil, "which", lambda _: None)

    async def go():
        async with tun.cloudflare_quick_tunnel(10000):
            pass

    import pytest

    with pytest.raises(tun.CloudflaredNotFound):
        asyncio.run(go())
