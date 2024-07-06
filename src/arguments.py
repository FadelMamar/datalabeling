
from dataclasses import dataclass
from typing import Sequence
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Arguments:

    # data preparation arguments
    root_path:str = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"
    dest_path_images:str=os.path.join(CUR_DIR,'../data/train/images')
    dest_path_labels:str=os.path.join(CUR_DIR,'../data/train/labels')
    ls_json_dir:str =os.path.join(CUR_DIR,"../exported_annotations/json")
    coco_json_dir:str = os.path.join(CUR_DIR,"../exported_annotations/coco-format")
    LABELSTUDIOCONFIG = os.path.join(CUR_DIR,"../exported_annotations/label_studio_config.xml")
    height:int = 640
    width:int = 640
    overlap:int=80
    min_visibility:float=0.1
    save_all:bool=False
    overlap_ratio:float=0.2
    empty_ratio:int=1

    # label mapping
    label_map:str=None

    # cli
    build_yolo_dataset:bool=False
    clear_yolo_dir:bool=False

    # run
    run_name:str='detector'

    # model type
    is_detector:bool=False
    
    # training data
    data_config_yaml:str=os.path.join(CUR_DIR,'../data/data_config.yaml')

    # labels to discard
    discard_labels:Sequence[str] = ('other','rocks','vegetation','detection','termite mound','label')

    # training details
    # path_weights:str=os.path.join(CUR_DIR,"../base_models_weights/yolov8m.pt") 
    path_weights:str=os.path.join(CUR_DIR,"../base_models_weights/yolov8.kaza.pt")
    lr0:float=1e-3
    lrf:float=1e-2
    batchsize:int=32
    epochs:int=50
    patience:int=10
    degrees:float=45.0
    flipud:float=0.0
    fliplr:float=0.5
    mosaic:float=0.0
    mixup:float=0.0
    erasing:float=0.0
    copy_paste:float=0.0
    shear:float=10.
    weightdecay:float=0.
    dropout:float=0
    multiscale:bool=False
    cos_lr:bool=False

    # model exporting format
    export_format:str=None
    export_batch_size:int=1
    export_model_weights:str=os.path.join(CUR_DIR,"../base_models_weights/yolov8.kaza.pt")

