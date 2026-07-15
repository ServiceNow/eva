# Pion probe — does a browser-class ICE stack fix the Telnyx WebRTC path?

Standalone Go program that drives the **exact** anonymous-WebRTC flow PR #173's
`TelnyxWebRTCClient` uses (`anonymous_login` → `telnyx_rtc.invite` → answer → media), but
with **Pion** instead of aiortc.

## Why

The assistant never hears EVA on the aiortc path (VSDK-277). Everything downstream — codec,
SRTP, region, framing, nomination mode — has been excluded. The browser works, and the one
thing it does differently is run **libwebrtc's ICE**. `aiortc` uses **aioice**, a pure-Python
ICE agent with known nomination deviations (it still does RFC 5245 *aggressive* nomination,
removed in RFC 8445). **Pion's ICE is a browser-class stack.** This probe answers one
question: does Pion's ICE get caller audio through b2bua-rtc where aioice doesn't?

If it prints `ASSISTANT HEARD US`, the fix is to move PR #173's media engine to Pion. If it
fails the same way, the problem is not the client ICE stack and the escalation stands.

## Run

Needs Go 1.21+. **No Telnyx account or API key** — anonymous login uses only the public
assistant id.

```bash
cd experiments/pion-probe
go mod tidy
go run . -assistant assistant-473093df-d97a-4eae-b484-413bcbe2fda4 -seconds 45
```

`speech.ogg` (48 kHz mono opus, committed) is streamed as the caller after the greeting.

Expected on success:

```
[ws] connected
[verto] session: <sessid>
[assistant] Thank you for calling Summit Health Plan prior authorization support...
[ice] connected
[audio] streaming caller speech
*** ASSISTANT HEARD US: "I'm calling from a provider office..." ***
==== RESULT ====
assistant heard us   : true (N user transcripts)
```

## Status

**Builds clean; not yet run against the assistant.** It was written and compiled in an
environment that blocks execution of freshly-built native binaries (both `go run` and a
plain external-linked `go build` binary are killed before `main` runs — this affects a
trivial `hello world` too, so it is a sandbox limitation, not a code issue). It needs to be
run on a normal developer machine to get the result. `go build ./...` passes.

## If it works: integrating Pion into PR #173

The swap boundary is small. `TelnyxWebRTCClient` (in `telnyx_server.py`) exposes
`start()`, `stop()`, `send_audio(pcm)`, `build_invite_params(sdp)`, an `audio_handler`
callback, and `disconnected_event`. Everything else in PR #173 (the EVA server contract,
Verto signaling, audio bridge, tool handling) talks only to that surface.

Two integration shapes:

1. **Pion as a media sidecar** (recommended): a small Go binary owns the peer connection;
   Python keeps the Verto signaling and exchanges SDP + PCM with it over a local Unix
   socket. `TelnyxWebRTCClient` becomes a thin wrapper that launches the sidecar. Python
   still drives `anonymous_login`/`invite`; Go produces the offer and consumes the answer.

2. **Pion owns signaling too**: move the whole client to Go, Python just spawns it and
   bridges audio + conversation events. Cleaner separation, more Go.

Either keeps PR #173 on the **anonymous `assistant_id`** path — no account, no ingress —
which is the hard constraint. This probe is shape (2) collapsed into one file, so it doubles
as the skeleton for the sidecar.
