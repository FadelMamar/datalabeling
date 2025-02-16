import scipy.stats
import yaml
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from datalabeling.annotator import Detector
import scipy


def load_prediction_results(path_result:str)->pd.DataFrame:
    return pd.read_json(path_result,orient='records')

# load groundtruth
def load_groundtruth(images_dir:str)->tuple[pd.DataFrame,list]:
    
    df_list = list()
    col_names = None
    num_empty = 0
    for image_path in Path(images_dir).glob('*'):

        # read label file and check if yolo or yolo-obb
        label_path = str(image_path.with_suffix('.txt')).replace('images','labels')

        # image is empty?
        if not os.path.exists(label_path):
            num_empty += 1
            continue

        df = pd.read_csv(label_path,sep=' ',header=None)
        if len(df.columns) == 9:
            df.columns = ['category_id','x1','y1','x2','y2','x3','y3','x4','y4']
        elif len(df.columns) == 5:
            df.columns = ['category_id','x','y','w','h']
        else:
            raise ValueError("Check features in label file.")
        
        # record features
        if col_names is None:
            col_names = list(df.columns)
        
        # add features
        df['image_path'] = str(image_path)
        width, height = Image.open(image_path).size
        df['width'] = width
        df['height'] = height

        # unnormalize values
        for i in range(1,5):
            df[f"x{i}"] = df[f"x{i}"]*width
            df[f"y{i}"] = df[f"y{i}"]*height        
        
        df_list.append(df)
    
    print(f"There are {num_empty} empty images in {images_dir}.")

    return pd.concat(df_list,axis=0), col_names

# get preds and targets
def get_preds_targets(images_dirs:list[str],pred_results_dir:str,detector:Detector,load_results:bool=False,save_tag:str=""):

    dfs_results = list()
    dfs_labels = list()
    features_names = None

    for image_dir in images_dirs:

        sfx = str(image_dir).split(":\\")[-1].replace("\\","_").replace("/","_")
        sfx = sfx + save_tag
        save_path = os.path.join(pred_results_dir,f'predictions-{sfx}.json')
        
        # get prediction results
        if load_results:
            results = load_prediction_results(save_path)
        else:
            results = detector.predict_directory(image_dir,as_dataframe=True,save_path=save_path)
        dfs_results.append(results)

        # get targets
        df_labels, col_names = load_groundtruth(images_dir=image_dir)
        dfs_labels.append(df_labels)

        # update and check for changes
        if features_names is None:
            features_names = col_names
        else:
            check_changes = len(set(features_names).intersection(set(col_names))) == len(features_names)
            assert check_changes, "groundtruth labels aren't all similar."
    
    return pd.concat(dfs_results,axis=0), pd.concat(dfs_labels,axis=0),features_names

# compute mAP@50
def compute_detector_performance(df_results:pd.DataFrame,df_labels:pd.DataFrame,col_names:list[str]):
    
    from torchmetrics.detection import MeanAveragePrecision

    m_ap = MeanAveragePrecision(box_format="xyxy",iou_type="bbox",
                                max_detection_thresholds=[1,10,100],
                                iou_thresholds=[0.15,0.25,0.35,0.5,0.75,0.85,0.95]
                                )

    def get_bbox(gt:np.ndarray):

        # empty image case
        if len(gt)<1:
            return np.array([])
        
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
    all_scores = list()
    # image_paths = list()
    image_paths = df_results['image_path'].unique()
    for image_path in tqdm(image_paths):

        # get gt
        mask_gt = df_labels['image_path'] == image_path
        gt = df_labels.loc[mask_gt,col_names].iloc[:,1:].to_numpy()
        labels = df_labels.loc[mask_gt,'category_id'].to_numpy().astype(int)

        # get preds
        mask_pred = df_results['image_path'] == image_path
        pred = df_results.loc[mask_pred,['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
        pred = np.clip(pred,a_min=0,a_max=pred.max())
        pred_score = df_results.loc[mask_pred,'score'].to_numpy()
        classes = df_results.loc[mask_pred,'category_id'].to_numpy().astype(int)
        max_scores.append(pred_score.max())
        all_scores.append(pred_score)

        # compute mAPs
        pred_list = [{'boxes':torch.from_numpy(pred),
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
                        "all_scores":all_scores,
                        "image_paths":image_paths}
    df_results_per_img = pd.DataFrame.from_dict(results_per_img,orient='columns')

    return df_results_per_img

# compute uncertainty
def get_uncertainty(df_results_per_img:pd.DataFrame,mode:str='entropy',reoder_ascending:bool=False):

    if mode == 'entropy':
        entropy_func = lambda x: -1*(np.log(x)*x).sum()
        df_results_per_img['uncertainty'] = df_results_per_img['all_scores'].apply(entropy_func)
        
    elif mode == '1-p':
        df_results_per_img['uncertainty'] = df_results_per_img['all_scores'].apply(lambda x: 1. - np.mean(x))
    
    else:
        raise NotImplementedError('mode is not implemented yet. entropy or 1-p')

    df_results_per_img.sort_values('uncertainty',axis=0,ascending=reoder_ascending,inplace=True)

    return df_results_per_img

# select images with low mAP@50 but high confidence
def select_hard_samples(df_results_per_img:pd.DataFrame,
                        score_col:str='max_scores',
                        map_thrs:float=0.3,
                        score_thrs:float=0.7,
                        save_path_samples:str=None,
                        root:str='D:\\',
                        uncertainty_method:str="entropy",
                        uncertainty_thrs:float=4,
                        save_data_yaml:str=None):
    """_summary_

    Args:
        df_results_per_img (pd.DataFrame): _description_
        score_col (str, optional): _description_. Defaults to 'max_scores'.
        map_thrs (float, optional): _description_. Defaults to 0.3.
        score_thrs (float, optional): _description_. Defaults to 0.7.
        save_path_samples (str, optional): _description_. Defaults to None.
        root (_type_, optional): _description_. Defaults to 'D:\'.
        uncertainty_method (str, optional): _description_. Defaults to "entropy".
        uncertainty_thrs (float, optional): _description_. Defaults to 4.
        save_data_yaml (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    df_results_per_img = get_uncertainty(df_results_per_img=df_results_per_img,mode=uncertainty_method)

    mask_low_map = (df_results_per_img['map50']<map_thrs) * (df_results_per_img['map75']<map_thrs)
    mask_high_scores = df_results_per_img[score_col]>score_thrs
    mask_low_scores =  df_results_per_img[score_col]< (1-score_thrs)
    mask_selected = mask_low_map * mask_high_scores + mask_low_map * mask_low_scores + (df_results_per_img['uncertainty']>uncertainty_thrs)
    df_hard_negatives = df_results_per_img.loc[mask_selected]

    # save image paths in data_config yaml
    if save_path_samples is not None:
        df_hard_negatives['image_paths'].to_csv(save_path_samples,index=False,header=False)
    if save_data_yaml is not None:
        cfg_dict = {'path':root,
                    'names': {0: 'wildlife'},
                    'train': os.path.relpath(save_path_samples,start=root),
                    'val':   os.path.relpath(save_path_samples,start=root),
                    'nc': 1,}
        with open(save_data_yaml,'w') as file:
            yaml.dump(cfg_dict,file)

    return df_hard_negatives



