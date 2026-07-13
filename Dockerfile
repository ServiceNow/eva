# Dockerfile for EVA
# Multi-stage build for smaller final image

# ============================================
# Stage 1: Deps — only rebuilds when pyproject.toml or uv.lock changes
# ============================================
FROM python:3.11-slim AS deps

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
# Stub src so uv can resolve project metadata without the real source
RUN mkdir -p src/eva && echo '__version__ = "0.0.0"' > src/eva/__init__.py
RUN uv venv /opt/venv && \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-cache

# ============================================
# Stage 2: Builder — reinstalls only the eva package on source changes
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Bring in the heavy deps venv from stage 1 (cached, only busts on uv.lock changes)
COPY --from=deps /opt/venv /opt/venv

# Copy real source and reinstall only the eva package (no deps, ~seconds)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps .

# ============================================
# Stage 3: Runtime
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies (ffmpeg, libsndfile1 for audio; curl for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy deps venv separately so it stays cached when only source changes
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Overlay only the eva package files that changed (tiny, ~seconds to copy)
COPY --from=builder /opt/venv/lib/python3.11/site-packages/eva /opt/venv/lib/python3.11/site-packages/eva
COPY --from=builder /opt/venv/lib/python3.11/site-packages/eva-*.dist-info /opt/venv/lib/python3.11/site-packages/
COPY --from=builder /opt/venv/bin/eva /opt/venv/bin/eva

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY assets/ ./assets/

# Create non-root user for runtime security
RUN groupadd --gid 1000 eva && \
    useradd --uid 1000 --gid eva --create-home eva

# Create directory for output with correct ownership
RUN mkdir -p /app/output && chown eva:eva /app/output

# Python runtime settings
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import eva; print('ok')" || exit 1

# Git provenance baked in at build time — placed LAST so that changing commit
# state only invalidates this tiny metadata layer, not the expensive deps/copy
# layers above it.
ARG GIT_COMMIT_SHA
ARG GIT_BRANCH
ARG GIT_DIRTY
ARG GIT_DIFF_HASH
ENV GIT_COMMIT_SHA=${GIT_COMMIT_SHA}
ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_DIRTY=${GIT_DIRTY}
ENV GIT_DIFF_HASH=${GIT_DIFF_HASH}

# Switch to non-root user
USER eva

ENTRYPOINT ["eva"]
