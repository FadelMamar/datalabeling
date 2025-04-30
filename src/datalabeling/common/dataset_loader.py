import json
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .annotation_utils import (
    ImageProcessor,
    LabelstudioConverter,
    load_coco_annotations,
)
from .config import DataConfig, LabelConfig, EvaluationConfig
from .io import load_yaml
from ..ml.models import Detector
from ..common.evaluation import PerformanceEvaluator

logger = logging.getLogger(__name__)


class LabelHandler:
    def __init__(self, config: LabelConfig):
        self.config = config

        self._label_map = None

        if (self.config.discard is not None) and (self.config.keep is not None):
            intersec = np.intersect1d(self.config.keep, self.config.discard)
            assert len(intersec) == 0, (
                f"{intersec} are required to be discarded and kept. Error..."
            )

    def load_map(
        self,
    ) -> Dict:
        """Load and filter label mapping"""

        # load label mapping
        with open(self.config.label_map, "r") as file:
            label_map = json.load(file)

        names = [p["name"] for p in label_map]

        if self.config.discard is not None:
            names = [p for p in names if p not in self.config.discard]

        if self.config.keep is not None:
            names = [p for p in label_map if p in self.config.keep]

        self._label_map = dict(zip(range(len(names)), names, strict=False))

        return label_map

    def update_config(self, yaml_path: Path) -> None:
        """Updates yolo data config yaml file "names" and "nc" fields."""

        # load yaml
        yolo_config = load_yaml(yaml_path)

        # updaate yaml and save
        yolo_config.update({"names": self._label_map, "nc": len(self._label_map)})
        with open(yaml_path, "w") as file:
            yaml.dump(yolo_config, file, default_flow_style=False, sort_keys=False)


class YOLODatasetBuilder:
    def __init__(
        self,
        data_config: DataConfig,
    ):
        self.config = data_config

        self._validate_config()

    def _validate_config(self):
        """Ensure configuration parameters are valid"""
        if self.config.slice_width <= 0 or self.config.slice_height <= 0:
            raise ValueError("Slice dimensions must be positive")

        #  Checking inconsistency in arguments
        if (self.config.clear_output + self.config.load_coco_annotations) == 2:
            raise ValueError(
                "Warning : both clear_yolo_dir and load_coco_annotations are enabled! "
                "it is likely to not work as expected."
            )

    def build(self, map_imgdir_cocopath: dict, label_handler: LabelHandler) -> None:
        """Main pipeline entry point"""

        # load label map and update yolo data_cfg_yaml file
        name_id_map = {}
        if not self.config.is_single_cls:
            label_map = label_handler.load_map()
            label_handler.update_config(self.config.yolo_data_config_yaml)
            name_id_map = {val: key for key, val in label_map.items()}

        # slice coco annotations and save tiles
        for img_dir, cocopath in tqdm(
            map_imgdir_cocopath.items(), desc="Building yolo dataset"
        ):
            try:
                # slice annotations
                coco_dict_slices = ImageProcessor.get_slices(
                    coco_path=cocopath, img_dir=img_dir, config=self.config
                )
                # sample tiles
                df_tiles = ImageProcessor.sample_slices(
                    coco_dict_slices=coco_dict_slices,
                    empty_ratio=self.config.empty_ratio,
                    out_csv_path=None,  # Path(args.dest_path_images).with_name("gt.csv"),
                    img_dir=img_dir,
                    save_all=self.config.save_all,
                    labels_to_discard=label_handler.config.discard,
                    labels_to_keep=label_handler.config.keep,
                    sample_only_empty=self.config.save_only_empty,
                )

                # detector_training mode
                if self.config.is_single_cls:
                    df_tiles["label_id"] = 0
                else:
                    df_tiles["label_id"] = df_tiles["labels"].map(name_id_map)
                    mask = ~df_tiles["label_id"].isna()
                    df_tiles.loc[mask, "label_id"] = df_tiles.loc[
                        mask, "label_id"
                    ].apply(int)

                # save labels in yolo format
                self.save_annotations(
                    df_annotation=df_tiles.dropna(axis=0, how="any"),
                    output_dir=self.config.dest_path_labels,
                )

                # save tiles
                ImageProcessor.save_tiles(
                    df_tiles=df_tiles,
                    output_dir=self.config.dest_path_images,
                    clear=self.config.clear_output,
                )

            except Exception:
                print("--" * 25, end="\n")
                traceback.print_exc()
                print("--" * 25)
                print(
                    f"Failed to build yolo dataset for for {img_dir} -- {cocopath}\n\n"
                )

    def save_annotations(self, df_annotation: pd.DataFrame, output_dir: Path) -> None:
        """Save annotations in YOLO format"""

        cols = ["label_id", "x", "y", "width", "height"]
        for col in cols:
            assert df_annotation[col].isna().sum() < 1, (
                "there are NaN values. Check out."
            )

        # change type
        for col in cols[1:]:
            df_annotation.loc[:, col] = df_annotation[col].apply(float)
        df_annotation.loc[:, "label_id"] = df_annotation["label_id"].apply(int)
        df_annotation = df_annotation.astype({"label_id": "int32"})

        # normalize values
        df_annotation.loc[:, "x"] = df_annotation["x"].apply(
            lambda x: x / self.config.slice_width
        )
        df_annotation.loc[:, "y"] = df_annotation["y"].apply(
            lambda y: y / self.config.slice_height
        )
        df_annotation.loc[:, "width"] = df_annotation["width"].apply(
            lambda x: x / self.config.slice_width
        )
        df_annotation.loc[:, "height"] = df_annotation["height"].apply(
            lambda y: y / self.config.slice_height
        )

        # check value range
        assert df_annotation[cols[1:]].all().max() <= 1.0, "max value <= 1"
        assert df_annotation[cols[1:]].all().min() >= 0.0, "min value >=0"

        for image_name, df in tqdm(
            df_annotation.groupby("images"), desc="Saving yolo labels"
        ):
            txt_file = image_name.split(".")[0] + ".txt"
            df[cols].drop_duplicates().to_csv(
                os.path.join(output_dir, txt_file), sep=" ", index=False, header=False
            )


class ClassificationDatasetBuilder:
    def __init__(
        self,
        detector: Detector,
        eval_config: EvaluationConfig,
        source_dir: str,
        output_dir: str,
    ):
        self.detector = detector
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.perf_eval = PerformanceEvaluator(config=eval_config)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    def save(
        self,
        image: np.ndarray,
        label_name: str | int,
        file_name: str,
        detection_index: int,
    ):
        img_dir = Path(self.output_dir) / str(label_name)
        img_dir.mkdir(exist_ok=True, parents=False)
        save_path = img_dir / f"{os.path.basename(file_name)}_det-{detection_index}.jpg"
        cv2.imwrite(save_path, image)

    def resize_bbox(self, factor: float, x1, x2, y1, y2, img_width, img_height):
        x1 = max(x1 - (factor - 1.0) * (x2 - x1) / 2, 0)
        y1 = max(y1 - (factor - 1.0) * (y2 - y1) / 2, 0)
        x2 = min(x2 + (factor - 1.0) * (x2 - x1) / 2, img_width)
        y2 = min(y2 + (factor - 1.0) * (y2 - y1) / 2, img_height)

        out = list(map(int, [x1, x2, y1, y2]))

        return out

    def process_images(self, bbox_resize_factor: float = 1.0):
        """Run batch detection and save cropped ROIs"""

        assert bbox_resize_factor >= 1.0

        df_metrics = self.perf_eval.evaluate(
            images_dirs=[self.source_dir],
            pred_results_dir=self.output_dir,
            save_tag="cls",
            detector=self.detector,
            load_results=False,
        )
        mask_fn = df_metrics["gt_FN"] == True
        mask_fp = df_metrics["pred_FP"] == True
        mask = mask_fn + mask_fp
        for file_name, df_det in tqdm(
            df_metrics.loc[mask, :].groupby("file_name"), desc="Saving FPs and FNs"
        ):
            image = Image.open(file_name).convert("RGB")
            img_width, img_height = image.size
            image = np.asarray(image)

            for i, row in df_det.iterrows():
                try:
                    x1 = int(row["pred_x_min"])
                    y1 = int(row["pred_y_min"])
                    x2 = int(row["pred_x_max"])
                    y2 = int(row["pred_y_max"])
                    # label_id = row['pred_category_id']
                    label_name = "false_positives"
                except:
                    x1 = int(row["gt_x_min"])
                    y1 = int(row["gt_y_min"])
                    x2 = int(row["gt_x_max"])
                    y2 = int(row["gt_y_max"])
                    # label_id = row['gt_category_id']
                    label_name = "false_negatives"

                x1, x2, y1, y2 = self.resize_bbox(
                    bbox_resize_factor, x1, x2, y1, y2, img_width, img_height
                )

                self.save(
                    image=image[y1:y2, x1:x2],
                    label_name=label_name,
                    file_name=file_name,
                    detection_index=i,
                )


class DataPreparation:
    def __init__(self, dataset_config: DataConfig, label_config: LabelConfig):
        self.dataset_config = dataset_config
        self.label_config = label_config

        self._initialize_components()

    def _initialize_components(self):
        self.label_handler = LabelHandler(self.label_config)
        self.converter = LabelstudioConverter(self.dataset_config)
        self.dataset_builder = YOLODatasetBuilder(self.dataset_config)

    def _clean_workspace(
        self,
    ):
        # clear directories
        if self.dataset_config.clear_output:
            for p in [
                self.dataset_config.dest_path_images,
                self.dataset_config.dest_path_labels,
                self.dataset_config.coco_json_dir,
            ]:
                shutil.rmtree(p)
                Path(p).mkdir(parents=True, exist_ok=True)
                logger.info(f"Deleting all content in: {p}")

    def run(self, ls_xml_config=None, ls_client=None, image_dir: str = None) -> None:
        """Execute full preparation pipeline"""

        # 1. Clean workspace
        self._clean_workspace()

        # 2. Convert source label studio format  to COCO
        if self.dataset_config.load_coco_annotations:
            map_imgdir_cocopath = load_coco_annotations(
                dest_dir_coco=self.dataset_config.coco_json_dir, image_dir=image_dir
            )
        else:
            map_imgdir_cocopath = self.converter.to_coco(
                input_dir=self.dataset_config.ls_json_dir,
                dest_dir_coco=self.dataset_config.coco_json_dir,
                parse_ls_config=self.dataset_config.parse_ls_config,
                ls_client=ls_client,
                ls_xml_config=ls_xml_config,
            )

        # 3. Load label map from ``label_handler.config.label_map``
        self.label_handler.load_map()

        # 3. Generate dataset
        self.dataset_builder.build(map_imgdir_cocopath, self.label_handler)
