import json
import logging
import os
import traceback
from pathlib import Path

import yaml
from datargs import parse
from dotenv import load_dotenv

from datalabeling.arguments import Dataprepconfigs
from datalabeling.dataset import (
    build_yolo_dataset,
    convert_obb_to_dota,
    convert_obb_to_yolo,
    convert_yolo_to_coco,
    convert_yolo_to_obb,
    create_yolo_seg_directory,
)


def load_yaml(data_config_yaml: str) -> dict:
    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    return data_config


def load_datasets(data_config_yaml: str) -> list[str]:
    data_config = load_yaml(data_config_yaml)
    paths = list()
    root = data_config["path"]
    for split in ["train", "val", "test"]:
        try:
            for p in data_config[split]:
                path = os.path.join(root, p)
                paths.append(path)
        except Exception as e:
            print(f"Failed to load datasets for conversion {split} --> ", e)

    return paths


if __name__ == "__main__":
    load_dotenv(r"..\.env")

    logger = logging.getLogger(__name__)

    args = parse(Dataprepconfigs)

    # print(args)

    # creates a yolo dataset given args and saves dataset creation configs
    if args.build_yolo_dataset:
        build_yolo_dataset(args=args)
        print("Saving arguments to destination directory")
        save_path = Path(args.dest_path_images).parent / "dataset_configs.json"
        with open(save_path, "w") as file:
            configs = [
                "is_detector",
                "discard_labels",
                "ls_json_dir",
                "keep_labels",
                "coco_json_dir",
                "dest_path_images",
                "dest_path_labels",
                "clear_yolo_dir",
                "height",
                "width",
                "overlap_ratio",
                "save_all",
                "parse_ls_config",
                "min_visibility",
                "empty_ratio",
            ]
            configs = dict(
                zip(configs, [args.__dict__[k] for k in configs], strict=False)
            )
            json.dump(configs, file, indent=2)

    assert (args.yolo_to_obb + args.obb_to_yolo) < 2, "Both arguments can't be True"

    # convert yolo dataset to obb
    if args.yolo_to_obb:
        paths = load_datasets(args.data_config_yaml)
        for p in paths:
            try:
                p_new = p.replace("images", "labels")
                logger.info(f"Converting {p_new}: yolo->obb")
                convert_yolo_to_obb(yolo_labels_dir=p_new, output_dir=p_new, skip=True)
            except Exception:
                logger.warning(f"Failed for {p_new}")
                traceback.print_exc()

    # convert obb dataset to yolo
    if args.obb_to_yolo:
        paths = load_datasets(args.data_config_yaml)
        for p in paths:
            try:
                p_new = p.replace("images", "labels")
                logger.info(f"Converting {p_new}: obb->yolo")
                convert_obb_to_yolo(obb_labels_dir=p_new, output_dir=p_new, skip=True)
            except Exception:
                logger.warning(f"Failed for {p_new}")
                traceback.print_exc()

    if args.obb_to_dota:
        data_config = load_yaml(args.data_config_yaml)

        for split in ["train", "val", "test"]:
            for img_dir in data_config[split]:
                try:
                    obb_img_dir = Path(data_config["path"], img_dir)
                    labels_output_dir = Path(obb_img_dir).parent / "dota_labels"
                    convert_obb_to_dota(
                        obb_img_dir=obb_img_dir,
                        output_dir=labels_output_dir,
                        label_map=data_config["names"],
                        skip=True,
                        clear_old_labels=args.clear_dota_labels,
                    )
                except Exception:
                    logger.warning(f"obb->dota failed for {obb_img_dir}")
                    traceback.print_exc()

    # create yolo-seg labels
    if args.create_yolo_seg_dir:
        from ultralytics import SAM

        model_sam = SAM(args.sam_model_path)
        create_yolo_seg_directory(
            data_config_yaml=args.data_config_yaml,
            model_sam=model_sam,
            device=args.device,
            copy_images_dir=args.copy_images,
        )

    if args.yolo_to_coco:
        from ultralytics.data.dataset import YOLOConcatDataset, YOLODataset

        from datalabeling.train.utils import remove_label_cache

        with open(args.data_config_yaml, "r") as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)

        remove_label_cache(args.data_config_yaml)

        for split in ["val", "test", "train"]:
            datasets = list()
            try:
                for path in data_config[split]:
                    images_path = os.path.join(data_config["path"], path)
                    dataset = YOLODataset(
                        img_path=images_path,
                        task="detect",
                        data={"names": data_config["names"]},
                        augment=False,
                        imgsz=args.imgsz,
                        classes=None,
                    )
                    datasets.append(dataset)
                datasets = YOLOConcatDataset(datasets)

                convert_yolo_to_coco(
                    datasets,
                    output_dir=args.coco_output_dir,
                    data_config=data_config,
                    split=split,
                    clear_data=args.clear_coco_dir,
                )
            except Exception as e:
                print(e)
                continue
