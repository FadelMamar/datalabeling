from arguments import Arguments
from datargs import parse
from utils import build_yolo_dataset


if __name__ == '__main__':

    args = parse(Arguments)

    if args.build_yolo_dataset:
        build_yolo_dataset(args=args,
                           clear_out_dir=args.clear_yolo_dir)