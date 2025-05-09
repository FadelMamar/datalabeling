from ml_microservices.inference_service.src.inference_service import (
    run_inference_server,
)
import os

if __name__ == "__main__":
    run_inference_server(port=os.environ["INFERENCE_PORT"])
