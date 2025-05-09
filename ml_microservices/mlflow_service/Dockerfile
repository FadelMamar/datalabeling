# 1. Use an official slim Python base
FROM python:3.11-slim-bookworm

# 2. Set MLflow and MinIO client versions
ENV MLFLOW_VERSION=2.22.0
ENV MINIO_CLIENT_VERSION=mc
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# 3. Install MLflow and AWS S3 extras for MinIO compatibility
RUN pip install mlflow[extras]==${MLFLOW_VERSION}  \
 && pip install boto3                        \
 && apt-get update && apt-get install -y --no-install-recommends curl  \
 && rm -rf /var/lib/apt/lists/*

# 4. Create a directory for MLflow metadata and artifacts
RUN mkdir -p /mlflow/mlruns
VOLUME ["/mlflow/mlruns"]

# 6. Expose the default MLflow port
EXPOSE 5000

# 7. Entrypoint: start the MLflow tracking server
CMD mlflow server --port 5000 --host 0.0.0.0 \
    --default-artifact-root http://minio:9000/mlflow-artifacts \
    --backend-store-uri /mlflow/mlruns
