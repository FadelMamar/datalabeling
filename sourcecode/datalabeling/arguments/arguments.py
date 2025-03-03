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
    data_config_yaml:str=None #os.path.join(CUR_DIR,'../../../data/data_config.yaml')    

    # labels to discard
    discard_labels:Sequence[str] = None #('other','rocks','vegetation','detection','termite mound','label')
    keep_labels:Sequence[str] = None

    # training flags
    height:int = 640
    width:int = 640
    imgsz:int = None
    path_weights:str=None #os.path.join(CUR_DIR,"../../../base_models_weights/yolov8m.pt") #os.path.join(CUR_DIR,"../../../base_models_weights/yolov8.kaza.pt")
    lr0:float=1e-4
    lrf:float=1e-2
    warmup_epochs:int=3
    batchsize:int=32
    epochs:int=50
    seed=41
    optimizer:str='AdamW'
    optimizer_momentum:float=0.99
    device:str= 0
    patience:int=10

    # pretraining
    use_pretraining:bool=False
    ptr_data_config_yaml:str=None
    ptr_tilesize:int=640
    ptr_batchsize:int=32
    ptr_epochs:int=10
    ptr_lr0:float=1e-4
    ptr_lrf:float=1e-1
    ptr_freeze:int=None

    # continual learning flags
    use_continual_learning:bool=False
    cl_ratios:Sequence[float]=(1.0,) # ratio = num_empty/num_non_empty
    cl_epochs:Sequence[int]=(20,)
    cl_freeze:Sequence[int]=(0,)
    cl_lr0s:Sequence[float]=(5e-5,)
    cl_save_dir:str=None # should be given!
    cl_data_config_yaml:str=None
    cl_batch_size:int=16

    # hard negative data sampling learning mode
    use_hn_learning:bool=False
    hn_save_dir:str=None
    hn_data_config_yaml:str=None
    hn_imgsz:int = 1280 # used to resize the input image
    hn_tilesize:int= 1280 # used for sliding window based detections
    hn_num_epochs:int=10
    hn_freeze:int=20
    hn_lr0:float=5e-5
    hn_lrf:float=1e-1
    hn_batch_size:int=16
    hn_is_yolo_obb:bool=False
    hn_use_sliding_window=True # can't change thru cli
    hn_overlap_ratio:float=0.2
    hn_map_thrs:float=0.35 # mAP threshold. lower than it is considered sample of interest
    hn_score_thrs:float=0.7
    hn_confidence_threshold:float=0.25
    hn_ratio:int=20 # ratio = num_empty/num_non_empty. Higher allows to look at all saved empty images
    hn_uncertainty_thrs:float=5 # helps to select those with high uncertainty
    hn_uncertainty_method:str="entropy"
    hn_load_results:bool=False

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
    scale:float=0.0
    fliplr:float=0.5
    flipud:float=0.5
    hsv_h:float=0.
    hsv_s:float=0.3
    hsv_v:float=0.3
    translate:float=0.2
    mosaic:float=0.

    # model exporting format
    export_format:str=None
    export_batch_size:int=1
    export_model_weights:str=os.path.join(CUR_DIR,"../../../base_models_weights/yolov8.kaza.pt")
    half:bool=False # to use FP16
    int8:bool=False # int8 quantization\
    dynamic:bool=False # allows dynamic input sizes 
    
@dataclass
class Dataprepconfigs:
    # data preparation arguments
    root_path:str = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"
    dest_path_images:str=None #os.path.join(CUR_DIR,'../../../data/train/images')
    dest_path_labels:str=None #os.path.join(CUR_DIR,'../../../data/train/labels')
    height:int=640
    width:int=640
    # overlap:int=80
    min_visibility:float=0.5
    save_all:bool=False
    overlap_ratio:float=0.1
    empty_ratio:float=0.1
    data_config_yaml:str=None #os.path.join(CUR_DIR,'../../../data/data_config.yaml')

    # annotations paths
    coco_json_dir:str=None
    ls_json_dir:str=None
    parse_ls_config:bool=False

    # convert dataset formats
    yolo_to_obb:bool=False
    obb_to_yolo:bool=False
    skip:bool=False # will skip wrongly formatted target files else, interrupts process
    
    # cli
    build_yolo_dataset:bool=False
    clear_yolo_dir:bool=False
    save_only_empty:bool=False
    
    # model type
    is_detector:bool=False

    # create coco from ls
    load_coco_annotations:bool=False

    # labels to discard
    discard_labels:Sequence[str] = None # ('other','rocks','vegetation','detection','termite mound','label')
    keep_labels:Sequence[str] = None

    # label mapping for identification dataset
    label_map:str = None

