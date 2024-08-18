from ultralytics import YOLO, RTDETR
from datargs import parse
from arguments import Arguments
import os
# import mlflow
# import wandb
import yaml

def main(args:Arguments):

    # Load a pre-trained model
    try:
    # if "yolo" in str(args.path_weights):
        model = YOLO(args.path_weights)
    # elif "rtdetr" in str(args.path_weights):
    except:
        model = RTDETR(args.path_weights)

    # Display model information (optional)
    model.info()

    # Remove labels.cache
    try:
        with open(args.data_config_yaml,'r') as file:
            yolo_config = yaml.load(file,Loader=yaml.FullLoader)
        root = yolo_config["path"]
        for p in yolo_config["train"] + yolo_config["val"]:
            os.remove(os.path.join(root,p,"../labels.cache"))
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
                cache=args.cache,
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


def run():

    args = Arguments()
    args.batchsize = 48
    args.lr0 = 5e-4
    args.lrf = 1e-2
    args.epochs = 250
    args.patience = 0
    args.optimizer = 'AdamW'
    args.cos_lr = True
    args.data_config_yaml = r"D:\PhD\Data per camp\Extra training data\WAID\data_config.yaml" #r"D:\PhD\Data per camp\IdentificationDataset\data_config.yaml"
    # args.dest_path_images = r"D:\PhD\Data per camp\IdentificationDataset\train\images" #r"D:\PhD\Data per camp\IdentificationDataset\train\images"
    args.project_name = 'Model trial on dataset WAID'
    args.weightdecay = 5e-4
    args.mosaic = 0.2
    args.mixup = 0.2
    args.degrees = 180.
    args.copy_paste = 0.3
    args.hsv_h = 0.015
    args.hsv_s = 0.4
    args.hsv_v = 0.4
    args.cache = True
    args.fliplr = 0.5
    args.flipud = 0.5
    args.shear = 10.
    args.multiscale = False
    


    # wandb.init()
    # config = wandb.config
    # args.path_weights = config.pathweights
    # args.lr0 = config.lr0

    for p in [
            #   "yolov5su",
            #   'yolov8s.pt',
              'yolov10s.pt'
              ]:
        # try:
        args.path_weights = p
        args.run_name = p
        # wandb.init(project=args.project_name,config=args,name=p)
        main(args=args)
        # except Exception as e:
        #     print(e,"\n")

# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "metrics/mAP50-95(B)"},
    "parameters": {
        "lr0": {"values": [1e-3,]},
        #"optimizer": {"values": ["Adam","SGD","AdamW"]},
        #"weightdecay" : {"values": [5e-2, 5e-3, 5e-4, 5e-5]},
        "pathweights" : {"values": [
                                    'yolov5su.pt',
                                     'yolov9s',
                                    'yolov10s.pt',
                                    'rtdetr-l.pt',
                                    'yolov8s.pt'
                                    ]}

    },
}

if __name__ == '__main__':
    run()
# 3: Start the sweep
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="Model trial on dataset v0")
    # wandb.agent(sweep_id, function=run, count=6)





