import yaml
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


def compute_predictions(weight_path:str,
                        task:str,
                        iou:float,
                        half:bool,
                        imgsz:int,
                        results_dir_name:str,
                        val_run_name:str,
                        data_config_yaml:str,
                        batch_size:int=1):
    
    # load model
    from ultralytics import YOLO
    model= YOLO(weight_path,task=task)

    # compute metrics per image
    _ = model.val(data=data_config_yaml,
                        project=results_dir_name,
                        name=val_run_name,
                        imgsz=imgsz,
                        iou=iou,
                        half=half,
                        save_json=True,
                        batch=batch_size)
    
    return os.path(results_dir_name,val_run_name,'predictions.json')

def load_prediciton_results(path_result:str):
    return pd.read_json(path_result,orient='records')

# load groundtruth
def load_groundtruth(data_config_yaml:str):

    with open(data_config_yaml,'r') as file:
        yolo_config = yaml.load(file,Loader=yaml.FullLoader)
    val_images_path = os.path.join(yolo_config['path'],yolo_config['val'][0])
    val_labels_path = val_images_path.replace('images','labels')

    df_list = list()
    col_names = None
    for path in Path(val_labels_path).glob('*.txt'):
        df = pd.read_csv(path,sep=' ',header=None)
        if len(df.columns) == 9:
            df.columns = ['id','x1','y1','x2','y2','x3','y3','x4','y4']
        elif len(df.columns) == 5:
            df.columns = ['id','x','y','w','h']
        else:
            raise ValueError("Check features in label file.")
        
        # record features
        if col_names is None:
            col_names = list(df.columns)

        df['image_id'] = path.stem
        image_path = os.path.join(val_images_path,f"{path.stem}.JPG")
        width, height = Image.open(image_path).size
        df['width'] = width
        df['height'] = height

        # unnormalize values
        for i in range(1,5):
            df[f"x{i}"] = df[f"x{i}"]*width
            df[f"y{i}"] = df[f"y{i}"]*height        
        
        df_list.append(df)

    return pd.concat(df_list,axis=0), col_names

# compute mAP@50
def compute_detector_performance(df_results:pd.DataFrame,df_labels,col_names:list[str]):
    
    from torchmetrics.detection import MeanAveragePrecision

    m_ap = MeanAveragePrecision(box_format="xyxy",iou_type="bbox",max_detection_thresholds=[10,100,300])

    def get_bbox(gt:np.ndarray):
        if len(col_names)==9:
            xs = [0,2,4,6]
            ys = [1,3,5,7]
            x_min = np.min(gt[:,xs],axis=1).reshape((-1,1))
            x_max = np.max(gt[:,xs],axis=1).reshape((-1,1))
            y_min = np.min(gt[:,ys],axis=1).reshape((-1,1))
            y_max = np.max(gt[:,ys],axis=1).reshape((-1,1))
        else:
            raise NotImplementedError("Support only yolo-obb outputs.")
        
        return np.hstack([x_min,y_min,x_max,y_max])

    map_50s = list()
    maps_75s = list()
    max_scores = list()
    image_paths = list()
    imgs_ids = df_results['image_id'].unique()
    for image_id in tqdm(imgs_ids):

        # get gt
        mask_gt = df_labels['image_id'] == image_id
        gt = df_labels.loc[mask_gt,col_names].iloc[:,1:].to_numpy()
        labels = df_labels.loc[mask_gt,'id'].to_numpy()
        p = df_labels.loc[mask_gt,'image_path'].unique()[0]
        p = str(p.with_suffix(".JPG")).replace('labels','images')
        image_paths.append(p)

        # get preds
        mask_pred = df_results['image_id'] == image_id
        pred = df_results.loc[mask_pred,'poly'].to_list()
        pred = np.array(pred)
        pred = np.clip(pred,a_min=0,a_max=pred.max())
        pred_score = df_results.loc[mask_pred,'score'].to_numpy()
        classes = df_results.loc[mask_pred,'category_id'].to_numpy()
        max_scores.append(pred_score.max())

        # compute mAPs
        pred_list = [{'boxes':torch.from_numpy(get_bbox(gt=pred)),
                'scores':torch.from_numpy(pred_score),
                'labels':torch.from_numpy(classes)}]
        target_list = [{"boxes":torch.from_numpy(get_bbox(gt=gt)),
                        "labels":torch.from_numpy(labels)}]

        
        metric = m_ap(preds=pred_list,target=target_list)
        map_50s.append(metric['map_50'].item())
        maps_75s.append(metric['map_75'].item())

    results_per_img = {"map50":map_50s,
                        "map75":maps_75s,
                        "max_scores":max_scores,
                        "image_ids":imgs_ids,
                        "image_paths":image_paths}
    df_results_per_img = pd.DataFrame.from_dict(results_per_img,orient='columns')

    return df_results_per_img

# select images with low mAP@50 but high confidence
def select_hard_samples(df_results_per_img:pd.DataFrame,
                        map_thrs:float=0.3,score_thrs:float=0.7,
                        save_path_samples:str=None,
                        root:str='D:\\',
                        save_data_yaml:str=None):
    
    mask_low_map = (df_results_per_img['map50']<map_thrs) * (df_results_per_img['map75']<map_thrs)
    mask_high_scores = df_results_per_img['max_scores']>score_thrs

    mask_selected = mask_low_map * mask_high_scores 
    df_hard_negatives = df_results_per_img.loc[mask_selected]

    # save image paths in data_config yaml
    if save_path_samples is not None:
        df_hard_negatives['image_paths'].to_csv(save_path_samples,
                                                sep=" ",index=False,header=False)
    if save_data_yaml is not None:
        cfg_dict = {'path':root,
                    'names': {0: 'wildlife'},
                    'train': os.path.relpath(save_path_samples,start=root),
                    'val':   os.path.relpath(save_path_samples,start=root),
                    'nc': 1,}
        with open(save_data_yaml,'w') as file:
            yaml.dump(cfg_dict,file)

    return df_hard_negatives