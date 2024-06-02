from ultralytics import YOLO
from datargs import parse
from arguments import Arguments

def main(args:Arguments):

    # Load a pre-trained model
    model = YOLO("yolov8.kaza.pt")

    # Display model information (optional)
    model.info()

    # Train the model
    model.train(data=args.data_config_yaml,
                epochs=args.epochs,
                imgsz=min(args.height,args.width),
                device=0,
                name='detector',
                single_cls=True,
                lr0=1e-3,
                lrf=1e-2,
                weight_decay=5e-4,
                dropout=0.2,
                batch=16,
                val=False,
                plots=True,
                cos_lr=True,
                deterministic=False,
                optimizer='Adam',
                project='wildAI',
                patience='10',
                degrees=45.0,
                mixup=0.2,
                shear=30.,
                exist_ok=True,
                seed=41
                )


if __name__ == '__main__':
    args = parse(Arguments)
    main(args=args)




