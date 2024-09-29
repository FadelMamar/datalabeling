from ultralytics import YOLO
import yaml
from ..arguments import Arguments
import os, logging, traceback

def start_training(args:Arguments):
    """Trains a YOLO model using ultralytics. By defaults, it will compile new 'labels.cache' files.

    Args:
        args (Arguments): configs
    """

    logger = logging.getLogger(__name__)

    # Load a pre-trained model
    model = YOLO(args.path_weights,task='detect',verbose=False)

    # Display model information (optional)
    model.info()    

    # Remove labels.cache
    try:
        with open(args.data_config_yaml,'r') as file:
            yolo_config = yaml.load(file,Loader=yaml.FullLoader)
        root = yolo_config["path"]
        for p in yolo_config["train"] + yolo_config["val"]:
            path = os.path.join(root,p,"../labels.cache")
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removing: {os.path.join(root,p,"../labels.cache")}")
    except Exception as e:
        # print(e)
        traceback.print_exc()

    # Train the model
    model.train(data=args.data_config_yaml,
                epochs=args.epochs,
                imgsz=min(args.height,args.width),
                device=args.device,
                freeze=args.freeze,
                name=args.run_name,
                single_cls=args.is_detector,
                lr0=args.lr0,
                lrf=args.lrf,
                momentum=args.optimizer_momentum,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                batch=args.batchsize,
                val=True,
                plots=True,
                cos_lr=args.cos_annealing,
                deterministic=False,
                optimizer=args.optimizer,
                project=args.project_name,
                patience=args.patience,
                multi_scale=False,
                degrees=args.rotation_degree,
                mixup=args.mixup,
                scale=args.scale,
                mosaic=args.mosaic,
                augment=False,
                erasing=args.erasing,
                copy_paste=args.copy_paste,
                shear=args.shear,
                fliplr=args.fliplr,
                flipud=args.flipud,
                perspective=0.,
                hsv_s=args.hsv_s,
                hsv_h=args.hsv_h,
                hsv_v=args.hsv_v,
                translate=args.translate,
                auto_augment='augmix',
                exist_ok=True,
                seed=args.seed
                )





