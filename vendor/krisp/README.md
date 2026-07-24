# Krisp VIVA SDK — build-time vendor files

The `krisp_viva_turn` turn-stop strategy needs Krisp's **licensed** VIVA SDK, which
is **not** on PyPI and **not** public — it is downloaded per-account from the Krisp
developer portal. The binaries are therefore **git-ignored** (see `.gitignore`) and
must be placed here manually before building the Docker image.

The image only publishes to our private Toolkit registry, so baking the licensed
files into the image is acceptable; committing them to git is not.

## What to put here

Download from https://developers.krisp.ai (log in with your Krisp account):

- **VIVA UAR Python SDK** zip → unzip → copy the **Linux x86_64** wheel matching the
  image's Python (3.11) into this directory:
  `krisp_audio-<ver>-cp311-cp311-linux_x86_64.whl`
  (The image is `linux/amd64` — do **not** use the macOS wheel you may have installed
  locally.)
- **Turn-Taking models** zip → unzip → copy the turn model here:
  `krisp-viva-tp-v3.kef`

Result:

```
vendor/krisp/
├── README.md                                          (tracked)
├── krisp_audio-1.10.0-cp311-cp311-linux_x86_64.whl    (git-ignored)
└── krisp-viva-tp-v3.kef                               (git-ignored)
```

## How it's used

The `Dockerfile` runtime stage installs the wheel and copies the `.kef` into the
image, setting `KRISP_VIVA_TURN_MODEL_PATH`. If this directory has no wheel, the build
still succeeds — Krisp is simply unavailable and `krisp_viva_turn` cannot be selected
at runtime.

`KRISP_VIVA_API_KEY` is **not** baked in — it is supplied as a runtime env var/secret,
like the other provider API keys.

## Runtime requirement

The SDK validates its license against `sdkapi.krisp.ai` and reports usage to
`analytics.krisp.ai` at runtime. The container must have outbound HTTPS to both, or
license validation fails.
