# Telnyx Media Streaming transport (auto-provisioned)

An alternative to the direct anonymous-WebRTC path for `EVA_FRAMEWORK=telnyx`. Instead of
opening a browser-style WebRTC session (ICE/DTLS/SRTP), EVA places a **Call Control**
outbound call to the assistant's SIP address and bridges audio over a **Media Streaming
WebSocket** (bidirectional PCMU). No WebRTC, so none of the ICE/gateway interop problems of
the WebRTC path apply.

**You provide only a Telnyx API key.** Everything else — the connection, an idle phone
number for caller ID, the outbound voice profile — is discovered or created for you.

> **Verified end to end.** A live EVA CLI run over this transport graded 1/1 (100%): the
> assistant heard the caller (2 user turns, real audio), and every metric — including the
> `user_speech_fidelity` audio judge — scored. See VSDK-277.

## Turnkey: `webhook_base_url: "auto"` (recommended)

Because EVA is one-shot, the transport can bring up its own **Cloudflare quick tunnel** for
the duration of the run — no account, no manual tunnel, no public URL to arrange. Set
`"webhook_base_url": "auto"` (or `"auto_tunnel": true`) in `s2s_params`; on start, EVA
launches `cloudflared` against its media port, uses the assigned `https://…trycloudflare.com`
URL, points the connection's webhook at it, and tears the tunnel down on exit. Requires the
`cloudflared` binary on PATH. Then it is just:

```bash
export TELNYX_API_KEY=KEY... OPENAI_API_KEY=sk-... GEMINI_API_KEY=AQ...
python -m eva.assistant.telnyx_provisioning --assistant-id assistant-473093df-... \
  --public-url https://placeholder --provision      # once: fills connection_id/from_number
uv run python main.py
```

with `EVA_MODEL__S2S_PARAMS='{"transport":"media_streaming","model":"…","assistant_id":"…","api_key":"KEY…","connection_id":"…","from_number":"…","webhook_base_url":"auto"}'`.

`GEMINI_API_KEY` is used by the audio judge (`user_speech_fidelity`), which calls Gemini
directly (not the LiteLLM router). `OPENAI_API_KEY` drives the Realtime caller + text judges.

The rest of this doc covers the manual-URL path (explicit tunnel or in-cluster ingress),
useful for a hosted/persistent deployment.

## 1. Provision (needs only the API key)

```bash
export TELNYX_API_KEY=KEY...

# read-only: shows what exists and what is missing, creates nothing
python -m eva.assistant.telnyx_provisioning \
  --assistant-id assistant-473093df-... \
  --public-url https://<your-public-url>

# create/attach whatever is missing (reuse-first; buys nothing)
python -m eva.assistant.telnyx_provisioning \
  --assistant-id assistant-473093df-... \
  --public-url https://<your-public-url> \
  --provision
```

It is **idempotent and reuse-first**: it looks for a Call Control Application named
`eva-media-streaming`, an existing outbound voice profile, and an **unattached** active
number (so it never disturbs the routing of a number already in use). Re-running is a no-op;
if your public URL changed (tunnels do), it re-points the connection's webhook.

It prints the `EVA_MODEL__S2S_PARAMS` to drop into `.env`:

```bash
EVA_FRAMEWORK=telnyx
EVA_MODEL__S2S=telnyx
EVA_MODEL__S2S_PARAMS='{"transport":"media_streaming","assistant_id":"assistant-473093df-...","api_key":"KEY...","connection_id":"3004642678097839809","from_number":"+13318719043","to":"sip:anonymous@assistant-473093df-...sip.telnyx.com","webhook_base_url":"https://<your-public-url>"}'
```

`telnyx_server` selects the media-streaming transport automatically whenever
`connection_id` and `webhook_base_url` are present in `s2s_params` (`_use_call_control`).

## 2. What "public ingress" means, and why you need it

The WebRTC path is **outbound-only**: EVA dials out and everything rides that one connection,
so EVA can sit behind NAT with no inbound reachability. **Media Streaming is different — it
requires Telnyx to connect back *to you*.** Two inbound flows:

1. **Webhooks** — after you start a call, Telnyx sends HTTP POSTs (`call.answered`,
   `streaming.started`, …) to `POST {webhook_base_url}/call-control-events`.
2. **The media WebSocket** — Telnyx opens a WebSocket *to your server* at
   `wss://{webhook_base_url}/media-stream/{conversation_id}` and streams the call audio over
   it, in both directions.

So Telnyx's servers, out on the public internet, must be able to **reach your EVA process**.
"Public ingress" is exactly that: a publicly resolvable HTTPS/WSS URL that routes inbound
connections to the local port EVA is listening on (default `:PORT` from the run config).

Your EVA process normally listens on `localhost` (or a private pod IP) that the public
internet cannot reach. You bridge that gap one of two ways:

- **A tunnel (easiest for a laptop or a locked-down pod).** A tool that opens an outbound
  connection to a public relay and gives you back a public URL that forwards to your local
  port:
  ```bash
  # cloudflared (no account needed for a quick tunnel)
  cloudflared tunnel --url http://localhost:8080
  #   -> https://random-words.trycloudflare.com   <- use this as --public-url

  # or ngrok
  ngrok http 8080
  #   -> https://xxxx.ngrok-free.app
  ```
  The tunnel must forward **both** HTTP (webhooks) and WebSocket (media) — cloudflared and
  ngrok both do. Note the URL changes each run unless you have a reserved/named tunnel; just
  re-run the provisioner to re-point the webhook.

- **A real ingress (for a deployment).** Expose the EVA service through a Kubernetes
  Ingress / load balancer with a public IP and a DNS name and TLS — e.g. a `Service` of
  type `LoadBalancer`, or an Ingress resource. Then `--public-url` is that stable hostname.

Whichever you pick, `webhook_base_url` in `s2s_params` must be that public URL, and it must
terminate TLS (Telnyx requires `https`/`wss`). The provisioner stores it on the connection's
webhook and it is used to build the media-stream URL handed to Telnyx.

## 3. Run

```bash
# EVA listens locally; the tunnel/ingress makes it reachable; the call streams audio back.
cloudflared tunnel --url http://localhost:8080        # -> copy the https URL
python -m eva.assistant.telnyx_provisioning --assistant-id ... --public-url <that-url> --provision
uv run python main.py
```

The call flow: EVA `POST /v2/calls` (connection_id, from, to=assistant SIP, `stream_url` =
your `wss://…/media-stream/…`) → Telnyx dials the assistant → `streaming.started` webhook →
Telnyx opens the media WebSocket to you → audio bridges over it, both directions.

## 4. Why this exists alongside the WebRTC path

| | WebRTC (`TelnyxWebRTCClient`) | Media Streaming (`TelnyxCallControlTransport`) |
|---|---|---|
| Credentials | public `assistant_id` only | API key (auto-provisions the rest) |
| Inbound reachability | **none needed** (outbound-only) | **public ingress required** |
| Media stack | ICE / DTLS / SRTP (aiortc) | plain WebSocket, PCMU — no ICE |
| Status | blocked on a b2bua-rtc ICE/media interop bug (VSDK-277) | not subject to that bug |

Use WebRTC when you can only supply an assistant id and can't expose a public URL (e.g. an
upstream EVA maintainer). Use Media Streaming when you have an API key and can provide public
ingress, and want a transport that sidesteps the WebRTC media path entirely.
