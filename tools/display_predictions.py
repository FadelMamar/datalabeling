import logging
import time
import traceback
from pathlib import Path

import fiftyone as fo
import numpy as np
import pandas as pd
from PIL import Image
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)
from sahi.utils.import_utils import check_requirements
from tqdm import tqdm

from datalabeling.annotator import Detector
from datalabeling.dataset.sampling import load_prediction_results

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}


LOW_MODEL_CONFIDENCE = 0.1


logger = logging.getLogger(__name__)


def load_groundtruth_fiftyone(target_path, is_yolo_obb: bool = True):
    def get_bbox(gt: np.ndarray):
        # empty image case
        if len(gt) < 1:
            return np.array([])

        if is_yolo_obb:
            xs = [0, 2, 4, 6]
            ys = [1, 3, 5, 7]
            x_min = np.min(gt[:, xs], axis=1).reshape((-1, 1))
            x_max = np.max(gt[:, xs], axis=1).reshape((-1, 1))
            y_min = np.min(gt[:, ys], axis=1).reshape((-1, 1))
            y_max = np.max(gt[:, ys], axis=1).reshape((-1, 1))
        else:
            raise NotImplementedError("Support only yolo-obb outputs.")

        return np.hstack([x_min, y_min, x_max, y_max])

    # load data
    df = pd.read_csv(target_path, sep=" ", header=None)
    if is_yolo_obb:
        df.columns = ["category_id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    else:
        df.columns = ["category_id", "x", "y", "w", "h"]

    gt = get_bbox(df.iloc[:, 1:].to_numpy())
    gt_detections = []
    for i in range(gt.shape[0]):
        x1, y1, x2, y2 = gt[i, 0], gt[i, 1], gt[i, 2], gt[i, 3]
        rel_box = [x1, y1, (x2 - x1), (y2 - y1)]
        gt_detections.append(fo.Detection(label="wildlife", bounding_box=rel_box))

    return gt_detections


def load_preds_fiftyone(path_to_results_json: str):
    df_results = load_prediction_results(path_to_results_json)

    pred_detections = dict()
    for filepath, df in tqdm(
        df_results.groupby("image_path"), desc="Loading preds from file"
    ):
        width, height = Image.open(filepath).size
        # normalize values
        df["x_min"] = df["x_min"] / width
        df["y_min"] = df["y_min"] / height
        df["bbox_w"] = df["bbox_w"] / width
        df["bbox_h"] = df["bbox_h"] / height
        # create detections
        cols = ["x_min", "y_min", "bbox_w", "bbox_h"]
        detections = [
            fo.Detection(
                label="wildlife",
                confidence=df["score"].iat[i],
                bounding_box=df[cols].iloc[i, :].to_list(),
            )
            for i in range(len(df))
        ]
        pred_detections[filepath] = detections

    return pred_detections, set(df_results["image_path"])


def predict_fiftyone(
    dataset_name: str,
    images_paths: list[str] | None,
    model_path: str,
    model_type: str = "mmdet",
    model_confidence_threshold: float = 0.25,
    model_device: str = None,
    eval_iou: float = 0.7,
    compute_preds: bool = True,
    load_resuts_from_path: str | None = None,
    uncertainty_mode: str = "entropy",
    use_sliding_window: bool = True,
    image_size: int = None,
    tilesize: int = 640,
    overlap_ratio: float = 0.15,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_threshold: float = 0.5,
    verbose: int = 1,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        model_type: str
            mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'.
        model_path: str
            Path for the model weight
        model_config_path: str
            Path for the detection model config file
        model_confidence_threshold: float
            All predictions with score < model_confidence_threshold will be discarded.
        model_device: str
            Torch device, "cpu" or "cuda"
        model_category_mapping: dict
            Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
        model_category_remapping: dict: str to int
            Remap category ids after performing inference
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        image_dir: str
            Folder directory that contains images or path of the image to be predicted.
        no_standard_prediction: bool
            Dont perform standard prediction. Default: False.
        no_sliced_prediction: bool
            Dont perform sliced prediction. Default: False.
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``256``.
        slice_width: int
            Width of each slice.  Defaults to ``256``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GREEDYNMM' or 'NMS'. Default is 'GREEDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices, model loading/file exporting durations
    """
    check_requirements(["fiftyone"])

    # for profiling
    durations_in_seconds = dict()

    if load_resuts_from_path is not None:
        print("INFO: images_paths is deduced from load_resuts_from_path.")
        assert not compute_preds, "compute_preds should be False"
        df_results, images_paths = load_preds_fiftyone(load_resuts_from_path)

    # create or load dataset
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(name=dataset_name)
    else:
        dataset = fo.Dataset(name=dataset_name)
        samples = []
        for filepath in images_paths:
            sample = fo.Sample(filepath=filepath)
            target_path = str(sample.filepath).replace("images", "labels")
            target_path = Path(target_path).with_suffix(".txt")
            sample["gt"] = fo.Detections(
                detections=load_groundtruth_fiftyone(target_path, is_yolo_obb=True)
            )
            samples.append(sample)
        dataset.add_samples(samples)
        dataset.save()

        dataset.persistent = True

    # init model instance
    time_start = time.time()
    if compute_preds:
        detection_model = Detector(
            path_to_weights=model_path,
            confidence_threshold=model_confidence_threshold,
            overlap_ratio=overlap_ratio,
            tilesize=tilesize,
            imgsz=image_size,
            device=model_device,
            use_sliding_window=use_sliding_window,
            is_yolo_obb=True,
        )

    # detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            if load_resuts_from_path is not None:
                sample[model_type] = fo.Detections(
                    detections=df_results[sample.filepath]
                )
                # print(df_results[sample.filepath])
                # exit()

            if compute_preds:
                # perform prediction
                prediction_result = detection_model.predict(
                    sample.filepath,
                    sahi_prostprocess=postprocess_type,
                    postprocess_match_threshold=postprocess_match_threshold,
                    return_coco=False,
                )
                # Save predictions to dataset
                sample[model_type] = fo.Detections(
                    detections=prediction_result.to_fiftyone_detections()
                )
                durations_in_seconds["slice"] += prediction_result.durations_in_seconds[
                    "slice"
                ]

            # TODO: add uncertainty to samples
            if uncertainty_mode:
                # get_uncertainty()
                pass

            sample.save()

        dataset.save()

    # print prediction duration
    if verbose == 1:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    # visualize results
    session = fo.launch_app(dataset=dataset)

    try:
        # Evaluate the predictions
        results = dataset.evaluate_detections(
            gt_field="gt",
            pred_field=model_type,
            classes=["wildlife"],
            eval_key="eval",
            iou=eval_iou,
            compute_mAP=True,
        )
        # Get the 10 most common classes in the dataset
        counts = dataset.count_values("gt.detections.label")
        classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
        # Print a classification report for the top-10 classes
        results.print_report(classes=classes_top10)
        # Load the view on which we ran the `eval` evaluation
        eval_view = dataset.load_evaluation_view("eval")
        # Show samples with most false positives
        session.view = eval_view.sort_by("eval_fp", reverse=True)

    except Exception:
        traceback.print_exc()

    while 1:
        time.sleep(3)


if __name__ == "__main__":
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv(r"../.env")

    # model_path = r'D:\datalabeling\models\best.pt'
    imgsz = 1280
    k = 1.5
    model_path = rf"D:\datalabeling\models\best_openvino_model_imgsz-{imgsz}"
    # images_paths = Path(r"D:\general_dataset\original-data\val\images").iterdir()
    dataset_name = "original-train"
    images_paths = None
    load_resuts_from_path = None  # r"D:\general_dataset\original-data\results\predictions-general_dataset_original-data_train_images_conf0.1-imgsz1280-tile2000-overlap0.1-sahiTrue.json"
    # images_paths = pd.read_csv(r"D:\general_dataset\original-data\results\hard_samples_train.txt",header=None).iloc[:,0].to_list()
    use_sahi = True
    predict_fiftyone(
        model_type="yolov8-obb",
        dataset_name=dataset_name,
        model_path=model_path,
        compute_preds=False,
        model_confidence_threshold=0.1,
        images_paths=images_paths,
        load_resuts_from_path=load_resuts_from_path,
        image_size=imgsz,
        tilesize=int(k * imgsz),
        eval_iou=0.5,
        use_sliding_window=use_sahi,
        postprocess_match_threshold=0.5,
        postprocess_type="NMS",
    )
