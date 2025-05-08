from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import ot  # Python Optimal Transport library


# =========================
# Dataset abstractions
# =========================
class BaseDataset(ABC):
    """
    Abstract parent class for datasets.
    Holds image paths, groundtruths, and predictions as DataFrames,
    tracks labeled/unlabeled indices, and requires update methods.
    """

    def __init__(self, image_paths: List[str]):
        # List of image file paths
        self.image_paths = image_paths
        # DataFrames to store annotations and detections
        self.groundtruth_df: pd.DataFrame = pd.DataFrame(columns=["image_idx", "box"])
        self.predictions_df: pd.DataFrame = pd.DataFrame(
            columns=["image_idx", "box", "score"]
        )
        # Track which images have labels vs unlabeled
        self.labeled_idx: List[int] = []
        self.unlabeled_idx: List[int] = list(range(len(image_paths)))

    @abstractmethod
    def split_labels(self):
        """Populate labeled_idx and unlabeled_idx based on current groundtruths."""
        pass

    @abstractmethod
    def update_groundtruths(self, groundtruth_df: pd.DataFrame):
        """Replace or merge new groundtruths into groundtruth_df and update splits."""
        pass

    @abstractmethod
    def update_predictions(self, predictions_df: pd.DataFrame):
        """Replace predictions_df with new detections."""
        pass

    def load_labeled_image_paths(self) -> List[str]:
        return [self.image_paths[i] for i in self.labeled_idx]

    def load_unlabeled_image_paths(self) -> List[str]:
        return [self.image_paths[i] for i in self.unlabeled_idx]


class SourceDataset(BaseDataset):
    def __init__(self, image_paths: List[str], groundtruth_df: pd.DataFrame):
        """
        image_paths: list of image file paths
        groundtruth_df: DataFrame with ['image_idx', 'box'] for each groundtruth
        """
        super().__init__(image_paths)
        self.groundtruth_df = groundtruth_df.copy()
        self.split_labels()

    def split_labels(self):
        labeled = self.groundtruth_df["image_idx"].unique().tolist()
        self.labeled_idx = labeled
        self.unlabeled_idx = [
            i for i in range(len(self.image_paths)) if i not in labeled
        ]

    def update_groundtruths(self, groundtruth_df: pd.DataFrame):
        self.groundtruth_df = groundtruth_df.copy()
        self.split_labels()

    def update_predictions(self, predictions_df: pd.DataFrame):
        self.predictions_df = predictions_df.copy()


class TargetDataset(BaseDataset):
    def __init__(self, image_paths: List[str]):
        super().__init__(image_paths)
        # initially unlabeled
        self.split_labels()

    def split_labels(self):
        labeled = self.groundtruth_df["image_idx"].unique().tolist()
        self.labeled_idx = labeled
        self.unlabeled_idx = [
            i for i in range(len(self.image_paths)) if i not in labeled
        ]

    def update_groundtruths(self, groundtruth_df: pd.DataFrame):
        # merge or replace
        self.groundtruth_df = groundtruth_df.copy()
        self.split_labels()

    def update_predictions(self, predictions_df: pd.DataFrame):
        self.predictions_df = predictions_df.copy()


class ActiveLearningDataset(BaseDataset):
    def __init__(self, source_dataset: SourceDataset, target_dataset: TargetDataset):
        # Empty list for abstract base init; not used directly
        super().__init__(image_paths=[])
        self.source = source_dataset
        self.target = target_dataset

    def split_labels(self):
        # No-op or synchronize both datasets if needed
        self.source.split_labels()
        self.target.split_labels()

    def update_groundtruths(self, groundtruth_df: pd.DataFrame, domain: str = "source"):
        if domain == "source":
            self.source.update_groundtruths(groundtruth_df)
        elif domain == "target":
            self.target.update_groundtruths(groundtruth_df)
        else:
            raise ValueError("Domain must be 'source' or 'target'")

    def update_predictions(self, predictions_df: pd.DataFrame, domain: str = "source"):
        if domain == "source":
            self.source.update_predictions(predictions_df)
        elif domain == "target":
            self.target.update_predictions(predictions_df)
        else:
            raise ValueError("Domain must be 'source' or 'target'")

    def get_labeled_source_paths(self) -> List[str]:
        return self.source.load_labeled_image_paths()

    def get_unlabeled_target_paths(self) -> List[str]:
        return self.target.load_unlabeled_image_paths()

    def get_groundtruths(self, domain: str = "source") -> pd.DataFrame:
        return (
            self.source.groundtruth_df
            if domain == "source"
            else self.target.groundtruth_df
        )

    def get_predictions(self, domain: str = "source") -> pd.DataFrame:
        return (
            self.source.predictions_df
            if domain == "source"
            else self.target.predictions_df
        )


class FeatureExtractor:
    def __init__(self, backbone: str = "resnet50"):
        # initialize CNN backbone
        pass

    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """Return array of shape (n_samples, feature_dim)"""
        return np.random.randn(len(crops), 512)


class DataPreprocessor:
    """Computes source features & labels, and target features"""

    def __init__(
        self,
        detector,
        feature_extractor: FeatureExtractor,
        evaluator: PerformanceEvaluator,
    ):
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.evaluator = evaluator

    def extract_source(self, dataset: SourceDataset) -> Tuple[np.ndarray, np.ndarray]:
        # Detect on labeled images
        image_paths = dataset.load_labeled_image_paths()
        raw_preds = self.detector.detect(image_paths)
        # Build predictions DataFrame
        preds = []
        for img_idx, dets in zip(dataset.labeled_idx, raw_preds):
            for box, score in dets:
                preds.append({"image_idx": img_idx, "box": box, "score": score})
        preds_df = pd.DataFrame(preds)
        dataset.update_predictions(preds_df)
        # Evaluate to get TP/FP
        perf_df = self.evaluator.evaluate(dataset, dataset.predictions_df)
        # Extract crops and features
        crops = perf_df["box"].tolist()
        feats = self.feature_extractor.extract(crops)
        labels = perf_df["is_tp"].astype(int).values
        return feats, labels

    def extract_target(self, dataset: TargetDataset) -> np.ndarray:
        image_paths = dataset.load_unlabeled_image_paths()
        raw_preds = self.detector.detect(image_paths)
        preds = []
        for img_idx, dets in zip(dataset.unlabeled_idx, raw_preds):
            for box, score in dets:
                preds.append({"image_idx": img_idx, "box": box, "score": score})
        preds_df = pd.DataFrame(preds)
        dataset.update_predictions(preds_df)
        crops = dataset.predictions_df["box"].tolist()
        feats = self.feature_extractor.extract(crops)
        return feats
