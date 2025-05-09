# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

# Install build-time dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*


# Prepare working directory
WORKDIR /build

# Copy only requirements for caching
COPY requirements.txt ./

# Install Python dependencies into /install
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN uv pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="fadel.seydou@gmail.com"
LABEL version="0.0.1"
LABEL description="Streamlit dashboard service using uv and multi-stage Docker build"

# Create application directory
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and environment file
COPY app.py ./
COPY utils.py ./

ENV LABEL_STUDIO_URL=http://labeling:8080
ENV INFERENCE_SERVICE_URL=http://inference:4141/predict

RUN mkdir -p /inference_results
VOLUME ["/inference_results"]

RUN uv run streamlit --version

# Expose the Streamlit port
EXPOSE 8501

# Launch Streamlit
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
