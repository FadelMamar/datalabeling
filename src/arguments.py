
from dataclasses import dataclass
from typing import Sequence
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Arguments:

    # data preparation arguments
    root_path:str = r'C:\Users\fadel'
    dest_path_images:str=os.path.join(CUR_DIR,'../data/train/images')
    dest_path_labels:str=os.path.join(CUR_DIR,'../data/train/labels')
    height:int = 640
    width:int = 640
    overlap:int=80
    min_visibility:float=0.1
    save_all:bool=False
    overlap_ratio:float=0.2
    empty_ratio:int=3

    # cli
    build_yolo_dataset:bool=False
    clear_yolo_dir:bool=False

    # model type
    is_detector:bool=True
    
    # training data
    data_config_yaml:str=os.path.join(CUR_DIR,'../data/data_config.yaml')

    # labels to discard
    discard_labels:Sequence[str] = ('other','rocks','vegetation','detection','termite mound','label')

    # training details
    path_weights:str=os.path.join(CUR_DIR,"../base_models_weights/yolov8m.pt") #os.path.join(CUR_DIR,"../base_models_weights/yolov8.kaza.pt")
    lr0:float=1e-3
    lrf:float=1e-2
    batchsize:int=16
    epochs:int=50

    # model exporting format
    export_format:str=None
    export_batch_size:int=1
    export_model_weights:str=os.path.join(CUR_DIR,"../base_models_weights/yolov8.kaza.pt")

