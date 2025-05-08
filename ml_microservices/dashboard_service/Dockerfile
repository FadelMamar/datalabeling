# Use an official slim Python base
FROM python:3.11-slim-bookworm

# Install your package and dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir -r requirements.txt

# Create & switch to app directory
WORKDIR /workspace

# 4. Install your package and dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 2. Create & switch to app directory
WORKDIR /app

# 3. Copy dependency files first (for better caching)
COPY .env app.py requirements.txt ./


# Expose the default port
EXPOSE 8501

# Entrypoint:
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
