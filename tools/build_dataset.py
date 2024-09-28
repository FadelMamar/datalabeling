from datalabeling.arguments import Dataprepconfigs
from datalabeling.dataset import build_yolo_dataset,convert_obb_to_yolo,convert_yolo_to_obb
from datargs import parse
from pathlib import Path
import json
import yaml
import os
import logging
import traceback

def load_datasets(data_config_yaml:str)->list[str]:

    with open(data_config_yaml,'r') as file:
        data_config = yaml.load(file,Loader=yaml.FullLoader)

    paths = list()
    root = data_config['path']
    for p in data_config['train']+data_config['val']:
        path = os.path.join(root,p)
        paths.append(path)

    return paths



if __name__ == '__main__':

    args = parse(Dataprepconfigs)

    # creates a yolo dataset given args and saves dataset creation configs
    if args.build_yolo_dataset:
        build_yolo_dataset(args=args)
        print("Saving arguments to destination directory")
        save_path = Path(args.dest_path_images).parent / "dataset_configs.json"
        with open(save_path,'w') as file:
            configs = ['is_detector',
                       'discard_labels','ls_json_dir',
                       'coco_json_dir','dest_path_images',
                       'dest_path_labels','clear_yolo_dir',
                       'height','width','overlap_ratio',
                       'empty_ratio']
            configs = dict(zip(configs,[args.__dict__[k] for k in configs]))
            json.dump(configs,file,indent=2)

    assert (args.yolo_to_obb + args.obb_to_yolo)<2, "Both arguments can't be True"

    # convert yolo dataset to obb
    if args.yolo_to_obb:
        paths = load_datasets(args.data_config_yaml)
        for p in paths:
            try:
                p_new = p.replace('images','labels')
                convert_yolo_to_obb(yolo_labels_dir=p_new,output_dir=p_new)
            except Exception as e:
                logging.warning(f"Failed for {p_new}")
                traceback.print_exc()

    # convert obb dataset to yolo
    if args.obb_to_yolo:
        paths = load_datasets(args.data_config_yaml)
        for p in paths:
            try:
                p_new = p.replace('images','labels')
                convert_obb_to_yolo(obb_labels_dir=p_new,output_dir=p_new)
            except Exception as e:
                logging.warning(f"Failed for {p_new}")
                traceback.print_exc()