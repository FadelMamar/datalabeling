import litserve as ls
import base64

from PIL import Image


class MyModelAPI(ls.LitAPI):
    def setup(
        self,
        device,
    ):
        """
        One-time initialization: load your model here.
        `device` is e.g. 'cuda:0' or 'cpu'.
        """

        from datalabeling.ml.models import Detector
        from datalabeling.common.annotation_utils import GPSUtils, ImageProcessor
        import torch

        self.model = Detector(
            path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
            confidence_threshold=0.25,
            overlap_ratio=0.2,
            tilesize=800,
            imgsz=800,
            use_sliding_window=True,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.get_image_gps_coord = GPSUtils.get_gps_coord
        self.get_px_gps_coord = ImageProcessor.generate_pixel_coordinates

    def decode_request(self, request: dict) -> Image.Image:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """
        from io import BytesIO

        try:
            img_data = request["image"]

            if not isinstance(img_data, str):
                raise ValueError("Invalid base64 format")

            image_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(image_bytes))  # .convert("RGB")

        except Exception as e:
            raise ValueError(f"Image decoding failed: {str(e)}")

        request["image"] = img
        request["image_gps"] = None
        request["return_coco"] = True

        # get gps coordinates
        return_as_decimal = request.get("return_as_decimal", False)
        gps, _ = self.get_image_gps_coord(
            file_name=None, image=img, return_as_decimal=return_as_decimal
        )
        request["image_gps"] = gps

        if "return_as_decimal" in request.keys():
            request.pop("return_as_decimal")

        request["return_gps"] = False

        return request

    def predict(self, x: dict):
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """
        import torch

        out = dict(image_gps=x.pop("image_gps"))

        with torch.no_grad():
            results = self.model.predict(**x)
            out["detections"] = results

        return out

    def encode_response(self, output: dict):
        """
        Wrap the model output in a JSON-serializable dict.
        """
        return output


# class MLflowModelFetcher:
#     """Downloads model artifacts from MLflow by run ID"""
#     def __init__(self, tracking_uri: str):
#         self.client = MlflowClient(tracking_uri)

#     def fetch(self, run_id: str, artifact_path: str, dst: str) -> str:
#         """Download the model artifact directory to dst, returns local path"""
#         return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=dst)


# def prepare_triton_model(repo_path: str, model_name: str, model_version: str, model_dir: str):
#     """
#     Copy the MLflow-fetched model to Triton's model repository structure:
#       {repo_path}/{model_name}/{model_version}/
#     """

#     dest = Path(repo_path) / model_name / model_version
#     if dest.exists(): shutil.rmtree(dest)
#     shutil.copytree(model_dir, dest)
#     print(f"Model staged at {dest}")


# @serve.deployment(route_prefix="/predict")
# @serve.ingress(FastAPI())
# class RayMNISTService:
#     def __init__(self, model_handle: RayServeHandle):
#         """Run inference using Triton gRPC client under the hood"""
#         from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
#         self.triton = InferenceServerClient(url="localhost:8001")
#         self.model_name = "my_model"
#         self.model_version = "1"

#     @serve.post("/")
#     async def predict(self, request: Any) -> Any:
#         data = await request.json()
#         # prepare Triton inputs
#         input_data = data["input"]  # e.g. list or nested lists
#         # build InferInput
#         input_tensor = InferInput(name="input__0", shape=list(data["shape"]), datatype="FP32")
#         input_tensor.set_data_from_numpy(np.array(input_data, dtype=np.float32))
#         outputs = [InferRequestedOutput(name="output__0")]
#         result = self.triton.infer(model_name=self.model_name, model_version=self.model_version, inputs=[input_tensor], outputs=outputs)
#         return {"predictions": result.as_numpy("output__0").tolist()}

# def start_rayserve():
#     """Bootstraps Ray Serve deployment after Triton is ready"""
#     model_fetcher = MLflowModelFetcher(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", ""))
#     local_dir = model_fetcher.fetch(run_id=os.environ.get("MLFLOW_RUN_ID"), artifact_path="model", dst="./models")
#     prepare_triton_model(repo_path="./triton_models", model_name="my_model", model_version="1", model_dir=local_dir)
#     serve.start(detached=True)
#     RayMNISTService.bind(None)
