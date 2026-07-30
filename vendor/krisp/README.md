# Krisp VIVA SDK — vendor files

The `krisp_viva_turn` turn-stop strategy needs Krisp's **licensed** VIVA SDK, which
is **not** on PyPI and **not** public — it is downloaded per-account from the Krisp
developer portal. The binaries are therefore **git-ignored** (see `.gitignore`) and
must be placed here manually before running the container.

## Download Krisp files

1. Log in to https://developers.krisp.ai/versions
2. Download the VIVA Python SDK zip file as well as the model zip files
3. Extract the necessary files
    ```sh
    cd vendor/krisp
    for file in ~/Downloads/krisp-viva-*-models*.zip; do unzip -jo "$file"; done
    for file in ~/Downloads/krisp-viva-uar-python-sdk-*.zip; do unzip -jo "$file" '*/dist/krisp_audio-*-cp311-cp311-linux_x86_64.whl'; done
    ```

Result (wheel version and `.kef` set will vary by download):

```
vendor/krisp/
├── README.md                                          (tracked)
├── krisp_audio-1.11.0-cp311-cp311-linux_x86_64.whl    (git-ignored)
└── krisp-viva-tp-v3.kef                               (git-ignored)
```

> [!WARNING]
> At most one `.whl` may be present here. The `eva` wrapper skips installing Krisp if it doesn't find one, and it errors out if it finds more than one, since it wouldn't know which to install.

## How it's used

In the Docker container:
- This directory is mounted to `/app/vendor/krisp`.
- The `eva` command (which is actually `scripts/docker_eva_wrapper.sh`) installs the `.whl`, if present.
- The `KRISP_VIVA_TURN_MODEL_PATH` env var points at the `.kef` file directly on that mount, e.g. `vendor/krisp/krisp-viva-tp-v3.kef`.
- The `KRISP_VIVA_API_KEY` env var provides the API key.

## Runtime requirement

The SDK validates its license against `sdkapi.krisp.ai` and reports usage to
`analytics.krisp.ai` at runtime. The container must have outbound HTTPS to both, or
license validation fails.
