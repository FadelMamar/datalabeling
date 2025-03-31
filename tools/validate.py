from datargs import parse
from dataclasses import dataclass
from typing import Sequence
import json

@dataclass
class Flags:
    
    # data_config file
    data_config: str = None

    # split
    splits:Sequence[str] = ("val",)

    # weights
    weights:str=None

    # inference
    batch_size: int = 32
    imgsz: int = 800
    iou_threshold: float = 0.6
    conf_threshold: float= 0.25
    device:str = "cuda"
    max_det:int = 50
    half=True
    augment:bool=False # Enables test-time augmentation (TTA) during validation, potentially improving detection accuracy at the cost of inference speed
    is_detector:bool=False

    # logging
    save_hybrid = False # Can cause false mAP ! used for autolabeling
    save_txt: bool = False # saves detection results in text files, with one file per image,
    save_json:bool = False # saves the results to a JSON file for further analysis, integration with other tools, or submission to evaluation servers like COCO.
    name:str = "val"
    project_name:str="wildAI-detection"
    plots:bool=False


def ultralytics_val(args:Flags):
    from ultralytics import YOLO

    # from pathlib import Path
    from datalabeling.train.utils import remove_label_cache

    
    

    # remove label.cache files
    remove_label_cache(data_config_yaml=args.data_config)

    for split in args.splits:
        print("-" * 20, split, "-" * 20)
        print('\n',args.weights,'\n',args.data_config,)
        model = YOLO(args.weights)
        model.info()

        # Customize validation settings
        name = args.name + "#" + split + f"#{round(args.conf_threshold*100)}#{round(args.iou_threshold*100)}#{args.augment}#{args.max_det}-"
        model.val(
            name=name,
            project=args.project_name,
            data=args.data_config,
            imgsz=args.imgsz,
            batch=args.batch_size,
            split=split,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            device=args.device,
            single_cls=args.is_detector,
            agnostic_nms=args.is_detector,
            augment=args.augment,
            save_conf=args.save_txt,
            save_crop=False,
            save_json=False,
            plots=args.plots,
            save_hybrid=args.save_hybrid,
            save_txt=args.save_txt
        )




if __name__ == "__main__":

    args = parse(Flags)

    print('\nargs:\n',json.dumps(args.__dict__,indent=2),'\n',flush=True)

    ultralytics_val(args)

