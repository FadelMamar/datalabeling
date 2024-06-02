
from dataclasses import dataclass

@dataclass
class Arguments:

    # data preparation arguments
    root_path:str = r'C:\Users\fadel'
    dest_path:str='../data/train/tiles_images'
    height:int = 512
    width:int = 512
    overlap:int=50
    min_visibility:float=0.0
    save_all:bool=False

    # model type
    is_detector:bool=True
