from datalabeling.arguments import Dataprepconfigs
from datalabeling.dataset import (
    build_yolo_dataset,
    convert_obb_to_yolo,
    convert_yolo_to_obb,
    create_yolo_seg_directory,
    convert_yolo_to_coco,
)
from datargs import parse
from pathlib import Path
import json
import yaml
import os
import logging
import traceback
from dotenv import load_dotenv


def load_datasets(data_config_yaml: str) -> list[str]:
    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

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
            configs = dict(zip(configs, [args.__dict__[k] for k in configs]))
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
        from ultralytics.data.dataset import YOLODataset, YOLOConcatDataset
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
                    clear_data=True,
                )
            except Exception as e:
                print(e)
                continue
