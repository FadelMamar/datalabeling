# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:29:12 2025

@author: FADELCO
"""

from tqdm import tqdm

from datalabeling.common.io import HerdnetData


def load_herd_net():
    data_config_yaml = r"D:\datalabeling\configs\yolo_configs\data_config.yaml"
    patch_size = 640
    batchsize = 4
    down_ratio = 2
    train_empty_ratio = 0.0

    datamodule = HerdnetData(
        data_config_yaml=data_config_yaml,
        patch_size=patch_size,
        tr_batch_size=batchsize,
        val_batch_size=1,
        down_ratio=down_ratio,
        train_empty_ratio=train_empty_ratio,
    )

    datamodule.setup("fit")
    num_classes = datamodule.num_classes

    for batch_train in tqdm(
        datamodule.train_dataloader(), desc="Iterating thru train_dataloader"
    ):
        continue

    for batch_val in tqdm(
        datamodule.val_dataloader(), desc="Iterating thru val_dataloader"
    ):
        continue


def load_classification_data():
    from datalabeling.common.config import EvaluationConfig
    from datalabeling.ml.models import Detector
    from datalabeling.common.dataset_loader import ClassificationDatasetBuilder

    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"

    detector = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        confidence_threshold=0.25,
        overlap_ratio=0.2,
        tilesize=800,
        imgsz=800,
        use_sliding_window=True,
        device="cpu",
    )

    handler = ClassificationDatasetBuilder(
        detector,
        eval_config,
        source_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images",
        output_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\cls",
    )

    handler.process_images(bbox_resize_factor=2.0)


if __name__ == "__main__":
    pass

    # load_classification_data()
