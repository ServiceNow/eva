"""Telnyx Media Streaming provisioning.

Given only a Telnyx API key, discover or create every account resource the Call Control
media-streaming transport needs to place a call to an AI Assistant:

  * a Call Control Application (the "connection" — carries the webhook URL and the
    outbound-calling settings),
  * an owned phone number to use as the caller ID (`from`), attached to that connection,
  * an outbound voice profile (so the connection is allowed to place calls).

Reuse-first and idempotent: nothing is created if a suitable resource already exists, and
re-running is a no-op. Read-only discovery needs no confirmation; creation is gated behind
`provision(create=True)`.
"""

from .provision import ProvisionResult, TelnyxProvisioner

__all__ = ["TelnyxProvisioner", "ProvisionResult"]
