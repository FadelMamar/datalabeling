# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

# Install build-time dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      sqlite3 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Prepare working directory
WORKDIR /build

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN uv pip install --no-cache-dir mlflow[extras]==2.22.0 boto3

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="fadel.seydou@gmail.com"
LABEL version="0.0.1"
LABEL description="MLflow tracking server (multi-stage) with MinIO backend"

# Create application directory
WORKDIR /mlflow

# Copy installed dependencies from builder
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Environment defaults
ENV BACKEND_STORE_URI=sqlite:///mlflow.db
ENV DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin


# Prepare volumes for persistence
RUN mkdir -p /mlflow/mlruns
VOLUME ["/mlflow"]

# 6. Expose the default MLflow port
EXPOSE 5000

# 7. Entrypoint: start the MLflow tracking server
CMD uv run mlflow server \
    --port 5000 \
    --host 0.0.0.0 \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --backend-store-uri ${BACKEND_STORE_URI} \
