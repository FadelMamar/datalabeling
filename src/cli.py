from arguments import Arguments
from datargs import parse


if __name__ == '__main__':

    args = parse(Arguments)

    if args.build_yolo_dataset:
        from utils import build_yolo_dataset
        build_yolo_dataset(args=args,
                           clear_out_dir=args.clear_yolo_dir)
        
    if args.export_format is not None:
        from ultralytics import YOLO
        model = YOLO(args.export_model_weights)
        assert args.width==args.height,'Input image should have a square shape.'
        model.export(format=args.export_format,
                     imgsz=args.width,
                     nms=True,
                     batch=args.export_batch_size,
                     simplify=True)