# Use an official slim Python base
FROM python:3.11-slim-bookworm

# Install your package and dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Create & switch to app directory
WORKDIR /app

# Copy dependency files first (for better caching)
COPY .env requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Expose the default port
EXPOSE 8501

# Entrypoint:
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
