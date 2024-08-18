from ultralytics import YOLO
import yaml
from datargs import parse
from ..arguments import Arguments
import os

def start_training(args:Arguments):

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
            os.remove(os.path.join(root,p,"../labels.cache"))
    except:
        pass

    # Train the model
    model.train(data=args.data_config_yaml,
                epochs=args.epochs,
                imgsz=min(args.height,args.width),
                device=args.device,
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
                multiscale=False,
                degrees=args.rotation_degree,
                mixup=args.mixup,
                scale=args.scale,
                mosaic=False,
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
    
    # TODO: export best weights to onnx or TensorRT
    # model = YOLO('../runs/detect/train/weight/best.pt')
    # model.export(format='onnx')


# if __name__ == '__main__':
#     args = parse(Arguments)
#     start_training(args=args)




