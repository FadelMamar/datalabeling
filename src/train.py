from ultralytics import YOLO
from datargs import parse
from arguments import Arguments

def main(args:Arguments):

    # Load a pre-trained model
    model = YOLO(args.path_weights)

    # Display model information (optional)
    model.info()

    # Train the model
    model.train(data=args.data_config_yaml,
                epochs=args.epochs,
                imgsz=min(args.height,args.width),
                device=0,
                name='detector',
                single_cls=True,
                lr0=args.lr0,
                lrf=args.lrf,
                weight_decay=5e-4,
                dropout=0.2,
                batch=args.batchsize,
                val=False,
                plots=True,
                cos_lr=True,
                deterministic=False,
                optimizer='Adam',
                project='wildAI',
                patience=10,
                degrees=45.0,
                mixup=0.2,
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




