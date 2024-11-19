import logging
import time
from pathlib import Path
import traceback

from datalabeling.annotator import Detector

from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)
from sahi.predict import get_sliced_prediction, get_prediction

from sahi.utils.import_utils import check_requirements

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

LOW_MODEL_CONFIDENCE = 0.1


logger = logging.getLogger(__name__)


def predict_fiftyone(
    model_type: str = "mmdet",
    model_path: str = None,
    model_config_path: str = None,
    model_confidence_threshold: float = 0.25,
    model_device: str = None,
    model_category_mapping: dict = None,
    model_category_remapping: dict = None,
    dataset_json_path: str = None,
    image_dir: str = None,
    compute_preds:bool=True,
    no_sliced_prediction: bool = False,
    image_size: int = None,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
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

    from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file, fo

    # for profiling
    durations_in_seconds = dict()

    # dataset = create_fiftyone_dataset_from_coco_file(image_dir, dataset_json_path)
    
    dataset_name = Path(image_dir).parent.name+"-"+Path(image_dir).name
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(name=dataset_name)
    else:
        dataset = fo.Dataset(name=dataset_name)
        samples = []
        for filepath in Path(image_dir).iterdir():
            sample = fo.Sample(filepath=filepath)
            samples.append(sample)
        dataset.add_samples(samples)
        dataset.save()
                                    
        dataset.persistent = True

    # init model instance
    time_start = time.time()
    detection_model = Detector(path_to_weights=model_path,
                                confidence_threshold=model_confidence_threshold,
                                overlap_ratio=overlap_height_ratio,
                                tilesize=min(slice_height,slice_width),
                                imgsz=image_size,
                                device=model_device,
                                use_sliding_window=True,
                                is_yolo_obb=True).detection_model
    
    # detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):

            if not compute_preds:
                break
            # perform prediction
            if not no_sliced_prediction:
                # get sliced prediction
                prediction_result = get_sliced_prediction(
                    image=sample.filepath,
                    detection_model=detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    perform_standard_pred=True,
                    postprocess_type=postprocess_type,
                    postprocess_match_threshold=postprocess_match_threshold,
                    postprocess_match_metric=postprocess_match_metric,
                    postprocess_class_agnostic=postprocess_class_agnostic,
                    verbose=verbose,
                )
                durations_in_seconds["slice"] += prediction_result.durations_in_seconds["slice"]
            else:
                # get standard prediction
                prediction_result = get_prediction(
                    image=sample.filepath,
                    detection_model=detection_model,
                    shift_amount=[0, 0],
                    full_shape=None,
                    postprocess=None,
                    verbose=0,
                )
                durations_in_seconds["prediction"] += prediction_result.durations_in_seconds["prediction"]

            # Save predictions to dataset
            sample[model_type] = fo.Detections(detections=prediction_result.to_fiftyone_detections())
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
    _ = fo.launch_app(dataset=dataset)
    # Evaluate the predictions
    # results = dataset.evaluate_detections(
    #     model_type,
    #     gt_field="ground_truth",
    #     eval_key="eval",
    #     iou=postprocess_match_threshold,
    #     compute_mAP=True,
    # )
    # # Get the 10 most common classes in the dataset
    # counts = dataset.count_values("ground_truth.detections.label")
    # classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
    # # Print a classification report for the top-10 classes
    # results.print_report(classes=classes_top10)
    # # Load the view on which we ran the `eval` evaluation
    # eval_view = dataset.load_evaluation_view("eval")
    # # Show samples with most false positives
    # session.view = eval_view.sort_by("eval_fp", reverse=True)
    while 1:
        time.sleep(3)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(r"../.env")

    use_sahi=False
    model_path = r'D:\datalabeling\models\best.pt'
    # model_path = r'D:\datalabeling\models\best_openvino_model'
    predict_fiftyone(model_type="yolov8-obb"+Path(model_path).stem,
                     model_path=model_path,
                     image_dir=r"D:\general_dataset\tiled-data\val\images",
                     postprocess_match_metric='IOU',
                     image_size=640,
                     slice_height=1280,
                     slice_width=1280,
                     compute_preds=True,
                     no_sliced_prediction = not use_sahi,
                    #  dataset_json_path=r"D:\general_dataset\tiled-data\val\sahi_preds.json",
                     postprocess_match_threshold=0.7,
                     postprocess_type='NMS'
                    )
    
