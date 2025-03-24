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

    # model_type
    model_type:str = "ultralytics" # ultralytics or HerdNet 


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


def herdnet_val():
    from datalabeling.train.herdnet import HerdnetData, HerdnetTrainer
    from datalabeling.arguments import Arguments
    import lightning as L
    import yaml
    import torch
    from lightning.pytorch.loggers import MLFlowLogger

    # lowering matrix multiplication precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    args = Arguments()
    args.imgsz = 800
    args.batchsize = 32
    down_ratio = 2

    # =============================================================================
    #     Identification
    # =============================================================================

    # args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    # checkpoint_path = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools\lightning-ckpts\epoch=11-step=1740.ckpt"

    # =============================================================================
    #     # Pretrained
    # =============================================================================
    # args.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"

    # =============================================================================
    #     Detection
    # =============================================================================
    args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification-detection.yaml"
    checkpoint_path = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\mlartifacts\934358897506090439\1f0f0be9be1a406c8df8978331a99915\artifacts\epoch=2-step=1815\epoch=2-step=1815.ckpt"

    # Get number of classes
    with open(args.data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    num_classes = data_config["nc"] + 1

    # set model
    mlf_logger = MLFlowLogger(
        experiment_name="Herdnet",
        run_name="herdnet-validate",
        tracking_uri=args.mlflow_tracking_uri,
        log_model=True,
    )
    herdnet_trainer = HerdnetTrainer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_config_yaml=args.data_config_yaml,
        lr=None,
        weight_decay=None,
        herdnet_model_path=None,
        loaded_weights_num_classes=num_classes,
        ce_weight=None,
        map_location="cpu",
        strict=True,
        work_dir="../.tmp",
    )

    # Data
    datamodule = HerdnetData(
        data_config_yaml=args.data_config_yaml,
        patch_size=args.imgsz,
        batch_size=args.batchsize,
        down_ratio=down_ratio,
        train_empty_ratio=0.0,
    )
    # Validation
    # datamodule.setup('fit')

    # Predict
    # images_path = os.path.join(data_config['path'],data_config['test'][0])
    # images_path = list(Path(images_path).glob('*'))
    # datamodule.set_predict_dataset(images_path=images_path,batchsize=1)

    trainer = L.Trainer(accelerator="auto", profiler="simple", logger=mlf_logger)

    # out = trainer.validate(model=herdnet_trainer,datamodule=datamodule)

    out = trainer.test(model=herdnet_trainer, datamodule=datamodule)

    # out = trainer.predict(model=herdnet_trainer, datamodule=datamodule,)

    print(out)


if __name__ == "__main__":

    args = parse(Flags)

    print('\nargs:\n',json.dumps(args.__dict__,indent=2),'\n',flush=True)

    ultralytics_val(args)

    # herdnet_val()

    pass
