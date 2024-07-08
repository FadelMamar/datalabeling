from ultralytics import YOLO, RTDETR
from datargs import parse
from arguments import Arguments
import os
# import mlflow
import wandb

def main(args:Arguments):

    # Load a pre-trained model
    try:
    # if "yolo" str(args.path_weights):
        model = YOLO(args.path_weights)
    # elif "rtdetr" in str(args.path_weights):
    except:
        model = RTDETR(args.path_weights)
    
    # Display model information (optional)
    model.info()    

    # Remove labels.cache
    try:
        CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        os.remove(os.path.join(args.dest_path_images,"../labels.cache"))
    except:
        pass

    # Train the model
    model.train(data=args.data_config_yaml,
                epochs=args.epochs,
                imgsz=min(args.height,args.width),
                device=0,
                name=args.run_name,
                single_cls=args.is_detector,
                iou=0.5,
                cache=False,
                augment=False, # For TTA predictions
                lr0=args.lr0,
                lrf=args.lrf,
                weight_decay=args.weightdecay,
                dropout=args.dropout,
                batch=args.batchsize,
                freeze=args.freeze,
                val=True,
                plots=True,
                cos_lr=args.cos_lr,
                deterministic=False,
                # multi_scale=args.multiscale,
                optimizer=args.optimizer,
                project=args.project_name,
                patience=args.patience,
                degrees=args.degrees,
                flipud=args.flipud,
                fliplr=args.fliplr,
                mosaic=args.mosaic,
                mixup=args.mixup,
                erasing=args.erasing,
                copy_paste=args.copy_paste,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
                scale=args.scale,
                perspective=0.,
                shear=args.shear,
                exist_ok=True,
                seed=41
                )
    

def run(config):

    args = Arguments()

    args.batchsize = 32
    args.lrf = 1e-3
    args.epochs = 50
    args.patience = 10
    args.optimizer = 'auto'
    args.data_config_yaml = "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml"
    args.dest_path_images = "D:\PhD\Data per camp\IdentificationDataset\train\images"

    
    with wandb.init(config=config):

        config = wandb.config
        args.path_weights = config.pathweights
        args.lr0 = config.lr0

        main(args=args)


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





