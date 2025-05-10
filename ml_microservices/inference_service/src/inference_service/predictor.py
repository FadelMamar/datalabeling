import litserve as ls
import base64, os

from PIL import Image
import torch
# from ray import serve

from .utils import (
    Detector,
    decode_request,
    postprocess_response,
    GPSUtils,
    ImageProcessor,
)


class MyModelAPI(ls.LitAPI):
    def setup(
        self,
        device,
    ):
        """
        One-time initialization: load your model here.
        `device` is e.g. 'cuda:0' or 'cpu'.
        """
        print("Device:", device)
        self.model = Detector(
            mlflow_model_name=os.environ.get("MODEL_NAME", "labeler"),
            mlflow_model_alias=os.environ.get("MODEL_ALIAS", "demo"),
            use_sliding_window=True,
            confidence_threshold=0.15,
            overlap_ratio=0.2,
            tilesize=960,
            imgsz=960,
            device=device,
            tracking_url=os.environ.get(
                "MLFLOW_TRACKING_URI", "http://mlflow_service:5000"
            ),
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
            img = Image.open(BytesIO(image_bytes))

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
            results = self.model.predict(
                **x,
            )
            out["detections"] = results

        return out

    def encode_response(self, output: dict):
        """
        Wrap the model output in a JSON-serializable dict.
        """
        return output


# @serve.deployment()
class DetectionService:
    def __init__(self):
        # One-time model load
        device = os.environ.get("DEVICE", "cpu")
        self.model = Detector(
            mlflow_model_name=os.environ["MODEL_NAME"],
            mlflow_model_alias=os.environ["MODEL_ALIAS"],
            use_sliding_window=True,
            confidence_threshold=0.15,
            overlap_ratio=0.2,
            tilesize=960,
            imgsz=960,
            device=device,
            tracking_url=os.environ["MLFLOW_TRACKING_URI"],
        )

    def detect(self, request_json):
        try:
            # Decode and prepare inputs
            inputs = decode_request(request_json)
        except Exception as e:
            # return serve.context.Response(
            #     response=f"Request decoding error: {e}", status_code=400
            # )
            return f"Request decoding error: {e}"

        # Run prediction
        try:
            with torch.no_grad():
                detections = self.model.predict(**inputs)
        except Exception as e:
            # return serve.context.Response(
            #     response=f"Model prediction error: {e}", status_code=500
            # )
            return f"Model prediction error: {e}"

        # Format and return response
        result = postprocess_response(inputs["image_gps"], detections)
        return result
