from zenml import step
from datalabeling.train import HerdnetData, HerdnetTrainer
from datalabeling.validation import ultralytics_validate, herdnet_validate

from datalabeling.arguments import Arguments
import lightning as L
import os, yaml
from pathlib import Path
import torch
import random
import pandas as pd
from lightning.pytorch.loggers import MLFlowLogger
# from ultralytics import YOLO
# from pathlib import Path
from datalabeling.train import remove_label_cache

@step
def ultralytics_val():
    
    # Getting results for yolov12s : Detection and Identification
    paths = ["../runs/mlflow/140168774036374062/e0ea49b51ce34cfe9de6b482a2180037/artifacts/weights/best.pt", # Identification model weights
            "../runs/mlflow/140168774036374062/a59eda79d9444ff4befc561ac21da6b4/artifacts/weights/best.pt" # Detection model weights
            ]

    dataconfigs = [
                    r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml",
                    r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification-detection.yaml"
                ]
    model_type='yolo'
    imgsz = 800
    iou_threshold=0.45
    conf_threshold=0.235
    batch=16
    device="cuda"
    splits = [
            "val", 
            "test",
            ]

    # remove label.cache files
    for dataconfig in dataconfigs:
        remove_label_cache(data_config_yaml=dataconfig)

    for split in splits:
        for path,dataconfig in zip(paths,dataconfigs):
            print("\n",'-'*20,split,'-'*20)
           ultralytics_validate(path, dataconfig, split, imgsz, 
                                conf_threshold, iou_threshold, model_type,
                                 device=device, batch=batch)

@step
def herdnet_val():

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
    with open(args.data_config_yaml, 'r') as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    num_classes = data_config['nc']+1
    
    
    # set model
    mlf_logger = MLFlowLogger(experiment_name="Herdnet",
                            run_name="herdnet-validate",
                            tracking_uri=args.mlflow_tracking_uri,
                            log_model=True
                            )

    herdnet_validate(...)
    

if __name__ == "__main__":

    # ultralytics_val()
    
    herdnet_val()
    

