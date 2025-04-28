import logging
import os

import numpy as np
import pandas as pd
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from ..ml.models import Detector
from .config import DataConfig, EvaluationConfig
from .io import DataHandler
from .selection import HardSampleSelector, UncertaintyAnalyzer

logger = logging.getLogger(__name__)


# =====================
# Performance Evaluation
# =====================
class PerformanceEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.label_format = None

    def evaluate(
        self, predictions: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate performance metrics"""
        metrics = self._calculate_base_metrics(predictions, ground_truth)
        return metrics

    def _calculate_base_metrics(
        self, df_pred: pd.DataFrame, df_gt: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute precision, recall, mAP etc."""

        m_ap = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 100],
            iou_thresholds=[0.15, 0.25, 0.35, 0.5, 0.75, 0.85, 0.95],
        )

        map_50s = list()
        maps_75s = list()
        max_scores = list()
        all_scores = list()

        image_paths = df_pred["file_name"].unique()
        for image_path in tqdm(image_paths, desc="Computing metrics"):
            # get gt
            mask_gt = df_gt["file_name"] == image_path
            gt = df_gt.loc[mask_gt, :].iloc[:, 1:].to_numpy()
            labels = df_gt.loc[mask_gt, "category_id"].to_numpy().astype(int)

            # get preds
            mask_pred = df_pred["file_name"] == image_path
            pred = df_pred.loc[
                mask_pred, ["x_min", "y_min", "x_max", "y_max"]
            ].to_numpy()
            pred = np.clip(pred, a_min=0, a_max=pred.max())
            pred_score = df_pred.loc[mask_pred, "score"].to_numpy()
            classes = df_pred.loc[mask_pred, "category_id"].to_numpy().astype(int)
            max_scores.append(pred_score.max())
            all_scores.append(pred_score)

            # compute mAPs
            pred_list = [
                {
                    "boxes": torch.from_numpy(pred),
                    "scores": torch.from_numpy(pred_score),
                    "labels": torch.from_numpy(classes),
                }
            ]
            target_list = [
                {
                    "boxes": torch.from_numpy(self._get_bbox(gt=gt)),
                    "labels": torch.from_numpy(labels),
                }
            ]

            metric = m_ap(preds=pred_list, target=target_list)
            map_50s.append(metric["map_50"].item())
            maps_75s.append(metric["map_75"].item())

        results_per_img = {
            "map50": map_50s,
            "map75": maps_75s,
            "max_scores": max_scores,
            "all_scores": all_scores,
            "file_name": image_paths,
        }

        return pd.DataFrame.from_dict(results_per_img, orient="columns")

    def _get_bbox(self, gt: np.ndarray):
        # empty image case
        if len(gt) < 1:
            return np.array([])

        if self.label_format == "yolo-obb":
            xs = [0, 2, 4, 6]
            ys = [1, 3, 5, 7]
            x_min = np.min(gt[:, xs], axis=1).reshape((-1, 1))
            x_max = np.max(gt[:, xs], axis=1).reshape((-1, 1))
            y_min = np.min(gt[:, ys], axis=1).reshape((-1, 1))
            y_max = np.max(gt[:, ys], axis=1).reshape((-1, 1))

        elif self.label_format == "yolo":
            x_min = (gt[:, 0] - gt[:, 2] / 2.0).reshape(-1, 1)
            x_max = (gt[:, 0] + gt[:, 2] / 2.0).reshape(-1, 1)
            y_min = (gt[:, 1] - gt[:, 3] / 2.0).reshape(-1, 1)
            y_max = (gt[:, 1] + gt[:, 3] / 2.0).reshape(-1, 1)

        else:
            raise NotImplementedError("label format should be yolo-obb or yolo.")

        return np.hstack([x_min, y_min, x_max, y_max]).astype(float)

    def get_preds_targets(
        self,
        images_dirs: list[str],
        pred_results_dir: str,
        detector: Detector,
        images_paths: list[str] = None,
        load_results: bool = False,
        save_tag: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # when providing a list of images
        if images_paths is not None:
            assert images_dirs is None, "images_dirs should be None!"
            sfx = save_tag
            save_path = os.path.join(pred_results_dir, f"predictions-{sfx}.json")

            # get prediction results
            if load_results:
                df_results = DataHandler.load_json_predictions(save_path)
            else:
                df_results = detector.predict_directory(
                    path_to_dir=None,
                    images_paths=images_paths,
                    as_dataframe=True,
                    return_gps=True,
                    save_path=save_path,
                )
            df_labels, label_format = DataHandler.load_yolo_groundtruth(
                images_dir=None, images_paths=images_paths
            )
            self.label_format = label_format

            return df_results, df_labels

        # when providing directories of images
        df_results = list()
        df_labels = list()
        labels_format = set()
        for image_dir in images_dirs:
            sfx = str(image_dir).split(":\\")[-1].replace("\\", "_").replace("/", "_")
            sfx = sfx + save_tag
            save_path = os.path.join(pred_results_dir, f"predictions-{sfx}.json")

            # get prediction results
            if load_results:
                results = DataHandler.load_json_predictions(save_path)
            else:
                results = detector.predict_directory(
                    path_to_dir=image_dir,
                    images_paths=None,
                    as_dataframe=True,
                    return_gps=True,
                    save_path=save_path,
                )
            df_results.append(results)

            # get targets
            labels, _format = DataHandler.load_yolo_groundtruth(
                images_dir=image_dir, images_paths=None
            )
            df_labels.append(labels)

            # update and check for changes
            labels_format.add(_format)
            (
                len(labels_format) == 1,
                f"There are inconcistencies in the labels formats.Found {labels_format}",
            )

        # Labels format
        self.label_format = labels_format.pop()

        return pd.concat(df_results, axis=0).reset_index(drop=True), pd.concat(
            df_labels, axis=0
        ).reset_index(drop=True)


# =====================
# Reporting Interface
# =====================
class ReportGenerator:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def generate_performance_report(self, metrics: pd.DataFrame) -> None:
        """Generate comprehensive performance report"""
        pass

    def generate_hard_samples_report(self, hard_samples: pd.DataFrame) -> None:
        """Generate report on challenging samples"""
        pass


# =====================
# Main Controller
# =====================
class CVModelEvaluator:
    def __init__(self, data_config: DataConfig, eval_config: EvaluationConfig):
        self.data_handler = DataHandler(data_config)
        self.evaluator = PerformanceEvaluator(eval_config)
        self.uncertainty = UncertaintyAnalyzer(eval_config)
        self.sample_selector = HardSampleSelector(eval_config)
        self.reporter = ReportGenerator(self.data_handler)

    def run_full_evaluation(
        self,
        detector: Detector,
        images_dirs: list[str],
        images_paths: list[str],
        pred_results_dir: str = None,
        save_tag: str = "",
        load_results: bool = False,
    ) -> None:
        """Complete evaluation workflow"""
        # Load data
        predictions, ground_truth = self.evaluator.get_preds_targets(
            detector=detector,
            images_dirs=images_dirs,
            images_paths=images_paths,
            pred_results_dir=pred_results_dir,
            load_results=load_results,
            save_tag=save_tag,
        )

        # Calculate metrics
        metrics = self.evaluator.evaluate(predictions, ground_truth)
        predictions = self.uncertainty.calculate_uncertainty(predictions)

        # Analyze results
        hard_samples = self.sample_selector.select_hard_samples(predictions)

        # Generate reports
        self.reporter.generate_performance_report(metrics)
        self.reporter.generate_hard_samples_report(hard_samples)

        return metrics, hard_samples
