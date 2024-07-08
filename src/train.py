from ultralytics import YOLO
from datargs import parse
from arguments import Arguments
import os
import yaml
import wandb

def main(args:Arguments):


    # Load a pre-trained model
    model = YOLO(args.path_weights)

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
                multi_scale=args.multiscale,
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

    # TODO: export best weights to onnx or TensorRT
    # model = YOLO('../runs/detect/train/weight/best.pt')
    # model.export(format='onnx')


if __name__ == '__main__':
    args = parse(Arguments)

    with wandb.init(project=args.project_name,
                     config=args,
                    name=args.run_name,
                    tags=args.tag):
        
        # log data_config file
        with open(args.data_config_yaml,'r') as file:
            data_config = yaml.load(file,Loader=yaml.FullLoader)
            wandb.log(data_config)

        main(args=args)





