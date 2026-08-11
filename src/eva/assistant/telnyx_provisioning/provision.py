"""Idempotent, reuse-first Telnyx account provisioning for the media-streaming transport.

Usage (read-only discovery — safe, no changes, no confirmation needed):

    python -m eva.assistant.telnyx_provisioning \
        --assistant-id assistant-473093df-... \
        --public-url https://your-tunnel.ngrok-free.app

Usage (create/attach missing resources — reuses everything it can, buys nothing):

    python -m eva.assistant.telnyx_provisioning \
        --assistant-id assistant-473093df-... \
        --public-url https://your-tunnel.ngrok-free.app \
        --provision

The API key is read from --api-key or TELNYX_API_KEY. On success it prints the
EVA_MODEL__S2S_PARAMS JSON to drop into your .env.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Self

import httpx

TELNYX_API = "https://api.telnyx.com/v2"

# Resources we own are tagged/named with this so re-runs find and reuse them.
MANAGED_APP_NAME = "eva-media-streaming"
MANAGED_PROFILE_NAME = "eva-media-streaming"


@dataclass
class ProvisionResult:
    connection_id: str
    from_number: str
    to: str
    webhook_base_url: str
    outbound_voice_profile_id: str | None = None
    created: list[str] | None = None

    def s2s_params(self, assistant_id: str, api_key: str) -> dict[str, Any]:
        """The EVA_MODEL__S2S_PARAMS payload for the media-streaming transport."""
        return {
            "transport": "media_streaming",
            "assistant_id": assistant_id,
            "api_key": api_key,
            "connection_id": self.connection_id,
            "from_number": self.from_number,
            "to": self.to,
            "webhook_base_url": self.webhook_base_url,
        }


class TelnyxProvisioner:
    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("A Telnyx API key is required (TELNYX_API_KEY or --api-key).")
        self._client = httpx.Client(
            base_url=TELNYX_API,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self._api_key = api_key

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- low-level helpers ---------------------------------------------------

    def _get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json().get("data", [])

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(path, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:400]}")
        return r.json().get("data", {})

    def _patch(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.patch(path, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"PATCH {path} -> {r.status_code}: {r.text[:400]}")
        return r.json().get("data", {})

    # --- discovery -----------------------------------------------------------

    def find_connection(self) -> dict[str, Any] | None:
        for app in self._get("/call_control_applications", **{"page[size]": 250}):
            if app.get("application_name") == MANAGED_APP_NAME:
                return app
        return None

    def find_outbound_profile(self) -> dict[str, Any] | None:
        for prof in self._get("/outbound_voice_profiles", **{"page[size]": 250}):
            if prof.get("name") == MANAGED_PROFILE_NAME:
                return prof
        # fall back to any existing profile so we don't create one unnecessarily
        existing = self._get("/outbound_voice_profiles", **{"page[size]": 1})
        return existing[0] if existing else None

    def list_owned_numbers(self) -> list[dict[str, Any]]:
        return self._get("/phone_numbers", **{"page[size]": 250, "filter[status]": "active"})

    def pick_from_number(self, preferred: str | None = None) -> dict[str, Any] | None:
        numbers = self.list_owned_numbers()
        if preferred:
            for n in numbers:
                if n.get("phone_number") == preferred:
                    return n
            raise ValueError(f"Requested from-number {preferred} is not active on this account.")
        # Prefer a number NOT already attached to a connection, so we never disrupt the
        # routing of a number that is in use elsewhere. Among those, prefer US/NANP.
        unattached = [n for n in numbers if not n.get("connection_id")]
        pool = unattached or numbers
        for n in pool:
            if str(n.get("phone_number", "")).startswith("+1"):
                return n
        return pool[0] if pool else None

    # --- provisioning --------------------------------------------------------

    def ensure(
        self,
        *,
        assistant_id: str,
        public_url: str,
        from_number: str | None = None,
        create: bool = False,
    ) -> ProvisionResult:
        created: list[str] = []
        webhook_base = public_url.rstrip("/")

        # 1) outbound voice profile (needed for the connection to place calls)
        profile = self.find_outbound_profile()
        if profile is None:
            if not create:
                raise _NeedsCreate("outbound_voice_profile", "no outbound voice profile exists")
            profile = self._post(
                "/outbound_voice_profiles",
                {"name": MANAGED_PROFILE_NAME, "traffic_type": "conversational"},
            )
            created.append(f"outbound_voice_profile:{profile['id']}")
        profile_id = profile["id"]

        # 2) Call Control Application (the connection), with our webhook URL
        conn = self.find_connection()
        webhook_event_url = f"{webhook_base}/call-control-events"
        if conn is None:
            if not create:
                raise _NeedsCreate(
                    "call_control_application",
                    f"no connection named {MANAGED_APP_NAME!r}",
                )
            conn = self._post(
                "/call_control_applications",
                {
                    "application_name": MANAGED_APP_NAME,
                    "webhook_event_url": webhook_event_url,
                    "outbound": {"outbound_voice_profile_id": profile_id},
                    "active": True,
                },
            )
            created.append(f"call_control_application:{conn['id']}")
        elif conn.get("webhook_event_url") != webhook_event_url:
            # keep the webhook pointed at the current public URL (tunnels change)
            if create:
                conn = self._patch(
                    f"/call_control_applications/{conn['id']}",
                    {"webhook_event_url": webhook_event_url},
                )
                created.append(f"call_control_application:{conn['id']}:webhook_updated")
        connection_id = conn["id"]

        # 3) a from-number attached to this connection
        number = self.pick_from_number(from_number)
        if number is None:
            raise RuntimeError(
                "No active phone number on the account to use as caller ID. "
                "Order one in the portal (Numbers -> Buy Numbers) or pass --from-number."
            )
        num_id = number["id"]
        if number.get("connection_id") != connection_id:
            if not create:
                raise _NeedsCreate(
                    "phone_number_assignment",
                    f"{number['phone_number']} is not attached to the connection",
                )
            self._patch(f"/phone_numbers/{num_id}", {"connection_id": connection_id})
            created.append(f"phone_number:{number['phone_number']}:attached")

        # the AI Assistant SIP destination — reachable with just the public assistant id
        to = f"sip:anonymous@{assistant_id}.sip.telnyx.com"

        return ProvisionResult(
            connection_id=connection_id,
            from_number=number["phone_number"],
            to=to,
            webhook_base_url=webhook_base,
            outbound_voice_profile_id=profile_id,
            created=created or None,
        )


class _NeedsCreate(RuntimeError):
    def __init__(self, resource: str, why: str) -> None:
        super().__init__(f"missing {resource}: {why} (re-run with --provision to create it)")
        self.resource = resource


def main() -> None:
    ap = argparse.ArgumentParser(description="Provision Telnyx resources for EVA media streaming.")
    ap.add_argument("--assistant-id", required=True)
    ap.add_argument("--public-url", required=True, help="Public base URL Telnyx can reach (tunnel or ingress).")
    ap.add_argument("--api-key", default=os.environ.get("TELNYX_API_KEY"))
    ap.add_argument("--from-number", default=None, help="E.164 number to use as caller ID (default: pick one).")
    ap.add_argument("--provision", action="store_true", help="Create/attach missing resources (else inspect only).")
    args = ap.parse_args()

    with TelnyxProvisioner(args.api_key) as prov:
        try:
            result = prov.ensure(
                assistant_id=args.assistant_id,
                public_url=args.public_url,
                from_number=args.from_number,
                create=args.provision,
            )
        except _NeedsCreate as exc:
            print(f"[inspect] {exc}")
            raise SystemExit(2) from exc

    print("\n=== provisioned ===")
    for k, v in asdict(result).items():
        print(f"  {k}: {v}")
    print("\n=== drop into .env ===")
    print("EVA_FRAMEWORK=telnyx")
    print("EVA_MODEL__S2S=telnyx")
    params = result.s2s_params(args.assistant_id, "<TELNYX_API_KEY>")
    print(f"EVA_MODEL__S2S_PARAMS='{json.dumps(params)}'")


if __name__ == "__main__":
    main()
