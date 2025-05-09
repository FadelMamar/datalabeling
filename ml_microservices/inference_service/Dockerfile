# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

# Install build-time tools
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml requirements.txt ./

# Install dependencies in a virtualenv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN uv pip install --upgrade pip \
 && uv pip install --no-cache-dir -e . \
 && uv pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="fadel.seydou@gmail.com"
LABEL version="0.0.1"
LABEL description="Inference service using litserve"

# Create app directory
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code only
COPY main.py ./
COPY src/ ./app/

# Create and mount model weights directory
RUN mkdir -p /model_weights
VOLUME ["/model_weights"]

# Expose inference port
EXPOSE 4141

# Entrypoint
CMD ["lightning", "serve", "main.py", "--local"]
