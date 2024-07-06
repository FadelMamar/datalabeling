from ultralytics import YOLO
from datargs import parse
from arguments import Arguments
import os
import mlflow

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
                cache=True,
                augment=False, # For TTA predictions
                lr0=args.lr0,
                lrf=args.lrf,
                weight_decay=args.weightdecay,
                dropout=args.dropout,
                batch=args.batchsize,
                val=True,
                plots=True,
                cos_lr=args.cos_lr,
                deterministic=False,
                multiscale=args.multiscale,
                optimizer='Adam',
                project='wildAI',
                patience=args.patience,
                degrees=args.degrees,
                flipud=args.flipud,
                fliplr=args.fliplr,
                mosaic=args.mosaic,
                mixup=args.mixup,
                erasing=args.erasing,
                copy_paste=args.copy_paste,
                shear=args.shear,
                exist_ok=True,
                seed=41
                )
    
    # TODO: export best weights to onnx or TensorRT
    # model = YOLO('../runs/detect/train/weight/best.pt')
    # model.export(format='onnx')


if __name__ == '__main__':
    args = parse(Arguments)
    main(args=args)




