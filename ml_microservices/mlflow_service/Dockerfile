# 1. Use an official slim Python base
FROM python:3.10-slim

# 2. Set MLflow and MinIO client versions
ENV MLFLOW_VERSION=2.5.0
ENV MINIO_CLIENT_VERSION=mc

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
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --host 0.0.0.0 --port 5000
