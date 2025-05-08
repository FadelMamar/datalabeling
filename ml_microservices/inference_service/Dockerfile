# Use an official slim Python base
# Base image
FROM python:3.11-slim-bookworm

# Install your package and dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Create & switch to app directory
WORKDIR /workspace

# Copy dependency files first (for better caching)
COPY pyproject.toml requirements.txt ./

RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir -r requirements.txt


 COPY main.py ./

# Copy your application code into /app
COPY app/ ./app/

# Create a directory for model weights and expose it
RUN mkdir -p /model_weights
VOLUME ["/model_weights"]

# Expose the default port
EXPOSE 4141

# Entrypoint:
CMD ["lightning", "serve", "main.py", "--local"]
