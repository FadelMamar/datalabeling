class MLflowModelFetcher:
    """Downloads model artifacts from MLflow by run ID"""

    def __init__(self, tracking_uri: str):
        self.client = MlflowClient(tracking_uri)

    def fetch(self, run_id: str, artifact_path: str, dst: str) -> str:
        """Download the model artifact directory to dst, returns local path"""
        return mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=dst
        )
