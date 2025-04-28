import litserve as ls


class MyModelAPI(ls.LitAPI):
    def setup(
        self,
        device,
        # path_to_weights,
        # confidence_threshold,
        # overlapratio,
        # tilesize
    ):
        """
        One-time initialization: load your model here.
        `device` is e.g. 'cuda:0' or 'cpu'.
        """

        from datalabeling.ml.models import Detector

        self.model = detector = Detector(
            path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
            confidence_threshold=0.25,
            overlap_ratio=0.2,
            tilesize=800,
            imgsz=800,
            use_sliding_window=True,
            device="cpu",
        )

    def decode_request(self, request: dict):
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """
        # e.g. for image bytes:
        from io import BytesIO

        from PIL import Image

        # import torchvision.transforms as T

        img_data = request["image_bytes"]
        img = Image.open(BytesIO(img_data)).convert("RGB")
        # transform = T.Compose([
        #     T.Resize((224, 224)),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        # ])
        # return transform(img).unsqueeze(0)  # add batch dim

        return img

    def predict(self, x):
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """
        import torch

        with torch.no_grad():
            outputs = self.model.predict(
                x, return_gps=False, return_coco=True, override_tilesize=False
            )
        # post-process to Python types
        # top_prob, top_label = torch.max(outputs, dim=1)
        # return {"label": top_label.item(), "confidence": top_prob.item()}
        return outputs

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
