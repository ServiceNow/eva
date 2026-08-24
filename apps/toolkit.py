"""Toolkit (EAI) helpers for the labeling app.

`get_user()` resolves the identity of the person using the app:

  * **Deployed on Toolkit** — the console's auth gateway authenticates the
    viewer (SSO) and attaches an ``Authorization`` header to every request it
    proxies into the job. We forward that header to ``{EAI_CONSOLE_URL}/v1/me``
    and let Toolkit resolve it to a user. ``EAI_CONSOLE_URL`` is injected into
    every Toolkit job automatically.
  * **Run locally** — there is no gateway, so we fall back to the logged-in
    ``eai`` CLI user (``eai user get``).

This mirrors the working implementation in the CLAE ``audio_rating`` app.
"""

from __future__ import annotations

import json
import os
import subprocess

import requests
import streamlit as st


def get_user() -> dict | None:
    """Return the current Toolkit user's info (dict with a ``mail`` field), or None.

    Fails closed: if the deployed auth header is missing/expired, the ``/v1/me``
    call is non-2xx and this returns None rather than guessing an identity.
    """
    eai_console_url = os.environ.get("EAI_CONSOLE_URL")
    if eai_console_url is None:
        # Running locally — use the logged-in eai CLI session.
        return json.loads(
            subprocess.check_output(("eai", "user", "get", "--format", "json"), text=True)
        )
    # Running on Toolkit — trust the identity the gateway attached to the request.
    authorization = st.context.headers.get("Authorization")
    if not authorization:
        return None
    response = requests.get(
        f"{eai_console_url}/v1/me",
        headers={"Authorization": authorization},
        timeout=1,
    )
    return response.json() if response.ok else None
