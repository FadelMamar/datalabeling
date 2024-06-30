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
                augment=False,
                iou=0.5,
                cache=True,
                lr0=args.lr0,
                lrf=args.lrf,
                weight_decay=0,
                dropout=0.2,
                batch=args.batchsize,
                val=True,
                plots=True,
                cos_lr=True,
                deterministic=False,
                optimizer='Adam',
                project='wildAI',
                patience=10,
                degrees=45.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=0.0,
                mixup=0.0,
                erasing=0.0,
                copy_paste=0.0,
                shear=10.,
                exist_ok=True,
                seed=41
                )
    
    # TODO: export best weights to onnx or TensorRT
    # model = YOLO('../runs/detect/train/weight/best.pt')
    # model.export(format='onnx')


if __name__ == '__main__':
    args = parse(Arguments)
    main(args=args)




