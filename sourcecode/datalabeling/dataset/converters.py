from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import shutil, os, yaml
from ultralytics import SAM
from ultralytics.data.dataset import YOLODataset, YOLOConcatDataset
import torch

def check_label_format(loaded_df: pd.DataFrame) -> str:
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
        raise NotImplementedError(
            f"The number of features ({num_features}) in the label file is wrong. Check yolo or yolo-obb format from ultralytics."
        )


def convert_yolo_to_obb(
    yolo_labels_dir: str, output_dir: str, skip: bool = True
) -> None:
    """Converts labels in yolo format to Oriented Bounding Box (obb) format.

    Args:
        yolo_labels_dir (str): directory with txt files following yolo format
        output_dir (str): output directory. It's a directory with txt files following guidelines at https://docs.ultralytics.com/datasets/obb/
    """

    cols = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    names = ["id", "x", "y", "w", "h"]

    if not Path(yolo_labels_dir).exists():
        raise FileNotFoundError("Directory does not exist.")

    # Iterate through labels
    for label_path in tqdm(Path(yolo_labels_dir).glob("*.txt"), desc="yolo->obb"):
        df = pd.read_csv(label_path, sep=" ", header=None)

        # check format is yolo
        if check_label_format(loaded_df=df) == "yolo":
            df.columns = names
            pass
        else:
            if skip:
                print(label_path, " does not follow yolo format. Skipped", end="\n")
                continue
            else:
                raise ValueError(f"{label_path} does not follow yolo format.")

        # check bounds
        assert df[names[1:]].all().max() <= 1.0, "max value <= 1"
        assert df[names[1:]].all().min() >= 0.0, "min value >=0"

        for col in names[1:]:
            df[col] = df[col].astype(float)
        df = df.astype({"id": "int32"})

        df["w"] = 0.5 * df["w"]
        df["h"] = 0.5 * df["h"]
        # top left
        df["x1"] = df["x"] - df["w"]
        df["y1"] = df["y"] - df["h"]
        # top right
        df["x2"] = df["x"] + df["w"]
        df["y2"] = df["y"] - df["h"]
        # bottom right
        df["x3"] = df["x"] + df["w"]
        df["y3"] = df["y"] + df["h"]
        # bottom left
        df["x4"] = df["x"] - df["w"]
        df["y4"] = df["y"] + df["h"]

        # check bounds
        assert df[names[1:]].all().max() <= 1.0, "max value <= 1"
        assert df[names[1:]].all().min() >= 0.0, "min value >=0"

        # save file
        df[cols].to_csv(
            Path(output_dir) / label_path.name, sep=" ", index=False, header=False
        )


def convert_obb_to_yolo(
    obb_labels_dir: str, output_dir: str, skip: bool = True
) -> None:
    """Converts labels in Oriented Bounding Box (obb) format to yolo format

    Args:
        obb_labels_dir (str): directory with txt files following guidelines at https://docs.ultralytics.com/datasets/obb/
        output_dir (str): output directory
    """

    names = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    cols = ["id", "x", "y", "w", "h"]

    if not Path(obb_labels_dir).exists():
        raise FileNotFoundError("Directory does not exist.")

    # Iterate through labels
    for label_path in tqdm(Path(obb_labels_dir).glob("*.txt"), desc="obb->yolo"):
        df = pd.read_csv(label_path, sep=" ", header=None)

        # check format
        if check_label_format(loaded_df=df) == "yolo-obb":
            df.columns = names
            pass
        else:
            if skip:
                print(label_path, " does not follow yolo-obb format. Skipped", end="\n")
                continue
            else:
                raise ValueError(f"{label_path} does not follow yolo-obb format.")

        # check bounds
        assert df[names[1:]].all().max() <= 1.0, "max value <= 1"
        assert df[names[1:]].all().min() >= 0.0, "min value >=0"

        # center
        df["x"] = (df["x1"] + df["x2"]) / 2.0
        df["y"] = (df["y1"] + df["y4"]) / 2.0
        # width
        df["w"] = df["x2"] - df["x1"]
        # height
        df["h"] = df["y4"] - df["y1"]

        # check bounds
        assert df[names[1:]].all().max() <= 1.0, "max value <= 1"
        assert df[names[1:]].all().min() >= 0.0, "min value >=0"

        # make sure id is int
        df = df.astype({"id": "int32"})

        # save file
        df[cols].to_csv(
            Path(output_dir) / label_path.name, sep=" ", index=False, header=False
        )


def convert_segment_masks_to_yolo_seg(masks_sam2:np.ndarray, output_path:str, num_classes:int,verbose:bool=False):
    """Inspired by https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py#L350
    Converts a SAM2 segmentation mask to the YOLO segmentation format.

    """
    assert len(masks_sam2.shape)==3, "[b,h,w]"
    pixel_to_class_mapping = {i + 1: i for i in range(num_classes)}

    file = open(output_path, "w", encoding="utf-8")
    for i in range(masks_sam2.shape[0]):
        mask = masks_sam2[i]
        img_height, img_width = mask.shape  # Get image dimensions

        unique_values = np.unique(mask)  # Get unique pixel values representing different classes
        yolo_format_data = []

        for value in unique_values:
            if value == 0:
                continue  # Skip background
            class_index = pixel_to_class_mapping.get(value, -1)
            if class_index == -1:
                print(f"Unknown class for pixel value {value}, skipping.")
                continue

            # Create a binary mask for the current class and find contours
            contours, _ = cv2.findContours(
                (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # Find contours

            for contour in contours:
                if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                    contour = contour.squeeze()  # Remove single-dimensional entries
                    yolo_format = [class_index]
                    for point in contour:
                        # Normalize the coordinates
                        yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                        yolo_format.append(round(point[1] / img_height, 6))
                    yolo_format_data.append(yolo_format)

        # Save Ultralytics YOLO format data to file
        
        for item in yolo_format_data:
            line = " ".join(map(str, item))
            file.write(line + "\n")
    if verbose:
        print(f"Processed and stored at {output_path}.")
    file.close()


def create_yolo_seg_directory(data_config_yaml:str,imgsz:int,model_sam:SAM,device:str='cpu',copy_images_dir:bool=True):

    with open(data_config_yaml, 'r') as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
        
    
    # Move to device
    model_sam = model_sam.to(device)
    print("Device:", model_sam.device)

    # get available splits in data_config
    splits=[s for s in ['val','train','test'] if s in data_config.keys()]
    print("Convertings splits:", splits)
    
    def is_dir_yolo(labels_dir:str)->None:
        
        for label_path in tqdm(Path(labels_dir).glob("*.txt"), desc="checking labels format"):
            df = pd.read_csv(label_path, sep=" ", header=None)
            if check_label_format(loaded_df=df) == "yolo":
                continue
            else:
                raise ValueError('Annotations should be in the yolo format')
        return None

    for split in splits:
        datasets = list()

        # Load YOLO dataset
        for path in data_config[split]:
            # create Segmentations directory inside split
            images_path = os.path.join(data_config['path'],path)
            
            # check folder format
            is_dir_yolo(images_path.replace('images', 'labels'))
            
            seg_dir = Path(images_path).parent/'Segmentations'
            seg_Labels_dir = seg_dir/'labels'
            if seg_Labels_dir.exists():
                shutil.rmtree(seg_Labels_dir)
                print(f"Deleting existing segmentation labels : {seg_Labels_dir}")
            seg_Labels_dir.mkdir(exist_ok=True,parents=True)
            if copy_images_dir:
                if (seg_dir/'images').exists():
                    shutil.rmtree(seg_dir/'images')
                    print('Deleting directory:',seg_dir/'images')
                shutil.copytree(images_path, seg_dir/'images')
                print(f'Copying {images_path} into {seg_dir}')
            dataset = YOLODataset(img_path=images_path,
                                  task='detect',
                                  data={'names':data_config['names']},
                                  augment=False,
                                  imgsz=imgsz,
                                  classes=None)
            datasets.append(dataset)
        dataset = YOLOConcatDataset(datasets)

        #  Saving segmentations
        for data in tqdm(dataset,desc=f"Creating yolo-seg for split={split}"):
            
            # skip negative samples
            if data['cls'].nelement() == 0:
                continue
            # Run inference with bboxes prompt
            bboxes = torch.cat([data['bboxes'][:,:2], data['bboxes'][:,:2] + data['bboxes'][:,2:]],1)
            bboxes = (bboxes*imgsz).long().tolist()
            results, = model_sam(data['im_file'],
                            imgsz=imgsz,
                            bboxes=bboxes,
                            labels=data['cls'].squeeze().long().tolist(),
                            device=device,
                            verbose=False
                            )
            # create masks
            mask = results.masks.data.cpu() * data['cls'].long().cpu().view(-1,1,1)
            assert len(mask.shape)==3
            # convert masks to yolo-seg
            img_path = Path(data['im_file'])
            output_dir = img_path.parent.parent/'Segmentations'/'labels'
            output_path = output_dir / img_path.with_suffix(".txt").name
            convert_segment_masks_to_yolo_seg(masks_sam2=mask.numpy(), 
                                                output_path=output_path, 
                                                num_classes=data_config['nc'],
                                                verbose=False
                                            )



