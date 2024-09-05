from datalabeling.arguments import Arguments, Dataprepconfigs
from datalabeling.dataset import build_yolo_dataset
from datargs import parse
from pathlib import Path
import json


if __name__ == '__main__':

    args = parse(Dataprepconfigs)

    if args.build_yolo_dataset:
        
        build_yolo_dataset(args=args)
        print("Saving arguments to destination directory")
        save_path = Path(args.dest_path_images).parent / "dataset_configs.json"
        with open(save_path,'w') as file:
            configs = ['is_detector',
                       'discard_labels','ls_json_dir',
                       'coco_json_dir','dest_path_images',
                       'dest_path_labels','clear_yolo_dir',
                       'empty_ratio']
            configs = dict(zip(configs,[args.__dict__[k] for k in configs]))
            json.dump(configs,file,indent=2)
    else:
        raise NotImplementedError