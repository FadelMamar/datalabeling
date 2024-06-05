
from dataclasses import dataclass

@dataclass
class Arguments:

    # data preparation arguments
    root_path:str = r'C:\Users\fadel'
    dest_path_images:str='../data/train/images'
    dest_path_labels:str='../data/train/labels'
    height:int = 640
    width:int = 640
    overlap:int=80
    min_visibility:float=0.0
    save_all:bool=False

    # model type
    is_detector:bool=True
    
    # training data
    data_config_yaml:str='../data/data_config.yaml'

    # training hyperparameters
    epochs:int=20

    # labels to discard
    discard_labels:tuple = ('other','rocks','vegetation','detection','termite mound','label')
