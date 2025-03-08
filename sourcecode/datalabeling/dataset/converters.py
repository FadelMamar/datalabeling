from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ..arguments.logger import logger
from zenml import step


def check_label_format(loaded_df:pd.DataFrame)->str:
    """checks label format

    Args:
        loaded_df (pd.DataFrame): target values

    Raises:
        NotImplementedError: when the format is not yolo or yolo-obb

    Returns:
        str: yolo or yolo-obb
    """

    num_features = len(loaded_df.columns)

    if num_features == 5:
        return "yolo"
    elif num_features == 9:
        return "yolo-obb"
    else:
        raise NotImplementedError(f"The number of features ({num_features}) in the label file is wrong. Check yolo or yolo-obb format from ultralytics.")

@step
def convert_yolo_to_obb(yolo_labels_dir:str,output_dir:str,skip:bool=True)->None:
    """Converts labels in yolo format to Oriented Bounding Box (obb) format.

    Args:
        yolo_labels_dir (str): directory with txt files following yolo format
        output_dir (str): output directory. It's a directory with txt files following guidelines at https://docs.ultralytics.com/datasets/obb/
    """

    cols = ['id','x1','y1','x2','y2','x3','y3','x4','y4']
    names = ['id','x','y','w','h']

    if not Path(yolo_labels_dir).exists():
        raise FileNotFoundError('Directory does not exist.')

    # Iterate through labels
    for label_path in tqdm(Path(yolo_labels_dir).glob("*.txt"),desc='yolo->obb'):
        df = pd.read_csv(label_path,sep=' ',header=None)
        
        # check format is yolo
        if check_label_format(loaded_df=df) == "yolo":
            df.columns = names
            pass
        else:
            if skip:
                logger.info(f"{label_path} does not follow yolo format. Skipped",end="\n")
                continue
            else:
                raise ValueError(f"{label_path} does not follow yolo format.")
                

        # check bounds
        assert df[names[1:]].all().max() <=1., "max value <= 1"
        assert df[names[1:]].all().min() >= 0., "min value >=0"

        for col in names[1:]:
            df[col] = df[col].astype(float)
        df = df.astype({'id':'int32'})

        df['w'] = 0.5*df['w']
        df['h'] = 0.5*df['h']
        # top left
        df['x1'] = df['x'] - df['w']
        df['y1'] = df['y'] - df['h']
        # top right
        df['x2'] = df['x'] + df['w']
        df['y2'] = df['y'] - df['h']
        # bottom right
        df['x3'] = df['x'] + df['w']
        df['y3'] = df['y'] + df['h']
        # bottom left
        df['x4'] = df['x'] - df['w']
        df['y4'] = df['y'] + df['h']

        # check bounds
        assert df[names[1:]].all().max() <=1., "max value <= 1"
        assert df[names[1:]].all().min() >= 0., "min value >=0"

        # save file
        df[cols].to_csv(Path(output_dir)/label_path.name,
                        sep=' ',index=False,header=False)

@step
def convert_obb_to_yolo(obb_labels_dir:str,output_dir:str,skip:bool=True)->None:
    """Converts labels in Oriented Bounding Box (obb) format to yolo format

    Args:
        obb_labels_dir (str): directory with txt files following guidelines at https://docs.ultralytics.com/datasets/obb/
        output_dir (str): output directory
    """

    names = ['id','x1','y1','x2','y2','x3','y3','x4','y4']
    cols = ['id','x','y','w','h']

    if not Path(obb_labels_dir).exists():
        raise FileNotFoundError('Directory does not exist.')

    # Iterate through labels
    for label_path in tqdm(Path(obb_labels_dir).glob("*.txt"),desc='obb->yolo'):
        df = pd.read_csv(label_path,sep=' ',header=None)

        # check format
        if check_label_format(loaded_df=df) == "yolo-obb":
            df.columns = names
            pass
        else:
            if skip:
                logger.info(f"{label_path} does not follow yolo-obb format. Skipped")
                continue
            else:
                raise ValueError(f"{label_path} does not follow yolo-obb format.")

        # check bounds
        assert df[names[1:]].all().max() <=1., "max value <= 1"
        assert df[names[1:]].all().min() >= 0., "min value >=0"

        # center
        df['x'] = (df['x1'] + df['x2'])/2.
        df['y'] = (df['y1'] + df['y4'])/2.
        # width
        df['w'] = df['x2'] - df['x1']
        # height
        df['h'] = df['y4'] - df['y1']

        # check bounds
        assert df[names[1:]].all().max() <=1., "max value <= 1"
        assert df[names[1:]].all().min() >= 0., "min value >=0"

        # make sure id is int
        df = df.astype({'id':'int32'})

        # save file
        df[cols].to_csv(Path(output_dir)/label_path.name,sep=' ',index=False,header=False)