from dataclasses import dataclass
from typing import Sequence
import os

# paths
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR_PATH = os.path.join(CUR_DIR,"../../../exported_annotations/json")
JSONMIN_DIR_PATH = os.path.join(CUR_DIR,"../../../exported_annotations/json-min")
CSV_DIR_PATH = os.path.join(CUR_DIR,"../../../exported_annotations/csv")
COCO_DIR_PATH = os.path.join(CUR_DIR,"../../../exported_annotations/coco-format")
ALL_CSV = os.path.join(CUR_DIR,"../../../exported_annotations/all_csv.csv")
LABELSTUDIOCONFIG = os.path.join(CUR_DIR,"../../../exported_annotations/label_studio_config.xml")
TEMP = os.path.join(CUR_DIR,"../../../.tmp")

@dataclass
class Arguments:

    # cli
    start_training:bool=False
       
    # model type
    is_detector:bool=False

    # active learning flags
    mlflow_tracking_uri:str="http://localhost:5000"
    mlflow_model_alias:str=None

    # training data
    data_config_yaml:str=os.path.join(CUR_DIR,'../../../data/data_config.yaml')    

    # labels to discard
    discard_labels:Sequence[str] = ('other','rocks','vegetation','detection','termite mound','label')

    # training flags
    path_weights:str=os.path.join(CUR_DIR,"../../../base_models_weights/yolov8m.pt") #os.path.join(CUR_DIR,"../../../base_models_weights/yolov8.kaza.pt")
    lr0:float=1e-4
    lrf:float=1e-2
    batchsize:int=32
    epochs:int=50
    seed=41
    optimizer:str='AdamW'
    optimizer_momentum:float=0.99
    device:str= 0
    patience:int=10

    # regularization
    dropout:float=0.
    weight_decay:float=5e-4

    # transfer learning
    freeze:int=None

    # lr scheduling
    cos_annealing:bool=True

    # run and project name MLOps
    run_name:str='detector'
    project_name:str='wildAI'
    tag:Sequence[str]=("",)
    
    # data augmentation https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
    rotation_degree:float=45.
    mixup:float=0.
    shear:float=10.
    copy_paste:float=0.
    erasing:float=0.
    scale:float=0.5
    fliplr:float=0.3
    flipud:float=0.3
    hsv_h:float=0.
    hsv_s:float=0.3
    hsv_v:float=0.4
    translate:float=0.1
    mosaic:float=0.

    # model exporting format
    export_format:str=None
    export_batch_size:int=1
    export_model_weights:str=os.path.join(CUR_DIR,"../../../base_models_weights/yolov8.kaza.pt")
    
@dataclass
class Dataprepconfigs:
    # data preparation arguments
    root_path:str = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"
    dest_path_images:str=os.path.join(CUR_DIR,'../../../data/train/images')
    dest_path_labels:str=os.path.join(CUR_DIR,'../../../data/train/labels')
    height:int = 640
    width:int = 640
    overlap:int=80
    min_visibility:float=0.1
    save_all:bool=False
    overlap_ratio:float=0.2
    empty_ratio:float=0.1
    data_config_yaml:str=os.path.join(CUR_DIR,'../../../data/data_config.yaml')

    # annotations paths
    coco_json_dir:str=""
    ls_json_dir:str=""

    # convert dataset formats
    yolo_to_obb:bool=False
    obb_to_yolo:bool=False
    
    # cli
    build_yolo_dataset:bool=False
    clear_yolo_dir:bool=False
    save_only_empty:bool=False
    
    # model type
    is_detector:bool=False

    # create coco from ls
    load_coco_annotations:bool=False

    # labels to discard
    discard_labels:Sequence[str] = ('other','rocks','vegetation','detection','termite mound','label')

    # label mapping for identification dataset
    label_map:str = None