from inference_service.endpoints import run_inference_server
import os


if __name__ == "__main__":
    run_inference_server(port=os.environ.get("INFERENCE_PORT", 4141))
