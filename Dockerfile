# Dockerfile for EVA
# Multi-stage build for smaller final image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies — cached as long as pyproject.toml doesn't change
COPY pyproject.toml README.md ./
# Stub src so hatchling can resolve the package during dep install
RUN mkdir -p src/eva && echo '__version__ = "0.0.0"' > src/eva/__init__.py
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy real source and reinstall only the package (deps already cached above)
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim AS runtime

# Git provenance baked in at build time
ARG GIT_COMMIT_SHA
ARG GIT_BRANCH
ARG GIT_DIRTY
ARG GIT_DIFF_HASH
ENV GIT_COMMIT_SHA=${GIT_COMMIT_SHA}
ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_DIRTY=${GIT_DIRTY}
ENV GIT_DIFF_HASH=${GIT_DIFF_HASH}

WORKDIR /app

# Install runtime dependencies (ffmpeg, libsndfile1 for audio; curl for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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

# Switch to non-root user
USER eva

ENTRYPOINT ["eva"]
