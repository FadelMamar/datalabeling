from ultralytics import YOLO, RTDETR
from datargs import parse
from arguments import Arguments
import os
# import mlflow
import wandb
from datalabeling.train import start_training

    

def run(config):

    args = Arguments()

    args.batchsize = 32
    args.lrf = 1e-3
    args.epochs = 50
    args.patience = 10
    args.optimizer = 'AdamW'
    args.data_config_yaml = "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml"
    args.dest_path_images = "D:\PhD\Data per camp\IdentificationDataset\train\images"

    
    with wandb.init(config=config):

        config = wandb.config
        args.path_weights = config.pathweights
        args.lr0 = config.lr0

        start_training(args=args)


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "metrics/mAP50-95(B)"},
    "parameters": {
        "lr0": {"values": [1e-3,]},
        #"optimizer": {"values": ["Adam","SGD","AdamW"]},
        #"weightdecay" : {"values": [5e-2, 5e-3, 5e-4, 5e-5]},
        "pathweights" : {"values": ['yolov5su.pt', "yolov6-s.pt",'yolov8s.pt','yolov9s.pt','yolov10-s.pt','rtdetr-l.pt']}

    },
}


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
wandb.agent(sweep_id, function=run, count=6)





