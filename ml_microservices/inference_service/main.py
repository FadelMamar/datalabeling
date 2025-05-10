from inference_service import run_inference_server, DetectionService
import os
import ray
from ray import serve

if __name__ == "__main__":
    # run_inference_server(port=os.environ.get("INFERENCE_PORT",
    #                                          4141)
    #                                         )

    ray.init(
        # address=os.environ.get("RAY_ADDRESS", ""),
    )

    # Start Serve runtime
    serve.start()

    # Deploy and expose route
    handle = serve.run(
        DetectionService.bind(), name="inference_service", blocking=False
    )
    serve.get_global_client().create_endpoint(
        name="predict",
        route="/predict",
        methods=["POST"],
        deployment="inference_service",
    )

    print("🚀 Ray Serve detection service deployed at POST /predict")
