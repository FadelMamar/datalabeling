"""Creates/gets an MLflow experiment and registers a detection model to the Model Registry."""

# import argparse
from dataclasses import dataclass
from sys import version_info

import cloudpickle
import mlflow
from datargs import parse

from datalabeling.mlflow import DetectorWrapper, get_experiment_id, model_wrapper


@dataclass
class Args:
    exp_name: str  # MLflow experiment name
    model: str  # Path to saved PyTorch model
    model_name: str  # Registered model name

    mlflow_tracking_uri: str = "http://localhost:5000"

    confidence_threshold: float = 0.1
    overlap_ratio: float = 0.1
    tilesize: int = 2000
    imgsz: int = 1280
    nms_iou: float = 0.5  # used when use_sliding_window=False

    use_sliding_window: bool = False

    is_yolo_obb: bool = False


PYTHON_VERSION = "{major}.{minor}.1".format(
    major=version_info.major, minor=version_info.minor
)

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python>=3.11",
        "pip",
        {
            "pip": [
                "mlflow>=2.13.2",
                "pillow",
                "ultralytics",
                "sahi",
                "cloudpickle",
                "torch>=2.0.0",
            ],
        },
    ],
    "name": "wildai_env",
}


def main():
    args = parse(Args)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    artifacts = {"path": args.model}

    model = DetectorWrapper(
        tilesize=args.tilesize,
        confidence_threshold=args.confidence_threshold,
        overlap_ratio=args.overlap_ratio,
        use_sliding_window=args.use_sliding_window,
        nms_iou=args.nms_iou,
        imgsz=args.imgsz,
        is_yolo_obb=args.is_yolo_obb,
        sahi_postprocess="NMS",
    )

    exp_id = get_experiment_id(args.exp_name)

    cloudpickle.register_pickle_by_value(model_wrapper)

    with mlflow.start_run(experiment_id=exp_id):
        mlflow.pyfunc.log_model(
            "finetuned",
            python_model=model,
            conda_env=conda_env,
            artifacts=artifacts,
            registered_model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
