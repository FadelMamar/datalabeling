from ultralytics import YOLO
import yaml
from ..arguments import Arguments
import os, logging, traceback
from pathlib import Path
import pandas as pd
import math


def sample_pos_neg(images_paths:list,ratio:float,seed:int=41):
    """_summary_

    Args:
        images_paths (list): _description_
        ratio (float): _description_
        seed (int, optional): _description_. Defaults to 41.

    Returns:
        _type_: _description_
    """

    # build dataframe
    is_empty = [1 - Path(str(p).replace('images','labels')).with_suffix('.txt').exists() for p in images_paths]
    data = pd.DataFrame.from_dict({"image_paths":images_paths,"is_empty":is_empty},
                                        orient="columns")
    # get empty and non empty
    num_empty = (data["is_empty"]==1).sum()
    num_non_empty = len(data)-num_empty
    if num_empty==0:
        print("contains only positive samples")
    num_sampled_empty = min(math.floor(num_non_empty*ratio),num_empty)
    sampled_empty = data.loc[data['is_empty']==1].sample(n=num_sampled_empty,random_state=seed)
    # concatenate
    sampled_data = pd.concat([sampled_empty,data.loc[data['is_empty']==0]])

    print(f"Sampling: pos={num_non_empty} & neg={num_sampled_empty}",end="\n")

    return sampled_data['image_paths'].to_list()


def get_data_cfg_paths_for_cl(ratio:float,data_config_yaml:str,cl_save_dir:str,seed:int=41,split:str='train',pattern_glob:str="*"):
    """_summary_

    Args:
        ratio (float): _description_
        data_config_yaml (str): _description_
        cl_save_dir (str): _description_
        seed (int, optional): _description_. Defaults to 41.
        split (str, optional): _description_. Defaults to 'train'.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    with open(data_config_yaml,'r') as file:
        yolo_config = yaml.load(file,Loader=yaml.FullLoader)

    root = yolo_config["path"]
    train_dirs_images = [os.path.join(root,p) for p in yolo_config[split]]
    
    # sample positive and negative images
    sampled_imgs_paths = []
    for dir_images in train_dirs_images:
        print(f"Sampling positive and negative samples from {dir_images}")
        paths = sample_pos_neg(images_paths=list(Path(dir_images).glob(pattern_glob)),
                       ratio=ratio,
                       seed=seed
                       )
        sampled_imgs_paths = sampled_imgs_paths + paths

    # save selected images in txt file
    save_path_samples = os.path.join(cl_save_dir,f"{split}_ratio_{ratio}-seed_{seed}.txt")
    pd.Series(sampled_imgs_paths).to_csv(save_path_samples,
                                        index=False,header=False)
    print(f"Saving {len(sampled_imgs_paths)} sampled images.")
    # save config
    if split == 'train':
        cfg_dict = {'path':root,
                    'names': yolo_config['names'],
                    'train': os.path.relpath(save_path_samples,start=root),
                    'val':   yolo_config['val'],
                    'nc': yolo_config['nc'],
                }
    elif split == 'val':
        cfg_dict = {'path':root,
                    'names': yolo_config['names'],
                    'val': os.path.relpath(save_path_samples,start=root),
                    'train':   yolo_config['val'],
                    'nc': yolo_config['nc'],
                }
    else:
        raise NotImplementedError
    save_path_cfg = Path(save_path_samples).with_suffix('.yaml')
    with open(save_path_cfg,'w') as file:
        yaml.dump(cfg_dict,file)

    print(f"Saving samples at: {save_path_samples} and data_cfg at {save_path_cfg}",end="\n\n")

    return str(save_path_cfg)


def get_data_cfg_paths_for_HN(args:Arguments, data_config_yaml:str):
    """_summary_

    Args:
        args (Arguments): _description_
        data_config_yaml (str): _description_

    Returns:
        _type_: _description_
    """

    from ..annotator import Detector
    from ..dataset.sampling import (get_preds_targets, compute_detector_performance,get_uncertainty)
    
    split="train"
    pred_results_dir=args.hn_save_dir
    save_path_samples= os.path.join(args.hn_save_dir,'hard_samples.txt')
    data_config_root="D:\\"
    save_data_config_yaml=os.path.join(args.hn_save_dir,'hard_samples.yaml')      

    # Define detector
    model = Detector(path_to_weights=args.path_weights,
                    confidence_threshold=args.hn_confidence_threshold,
                    overlap_ratio=args.hn_overlap_ratio,
                    tilesize=args.hn_tilesize,
                    imgsz=args.hn_imgsz,
                    use_sliding_window=args.hn_use_sliding_window,
                    device=args.device,
                    is_yolo_obb=args.hn_is_yolo_obb
            )

    # data config yaml
    with open(data_config_yaml,'r') as file:
        yolo_config = yaml.load(file,Loader=yaml.FullLoader)
    # get images_paths
    images_paths = os.path.join(yolo_config['path'],yolo_config[split])
    images_paths = pd.read_csv(images_paths,header=None,names=['paths'])['paths'].to_list()
    # get predictions & targets
    df_results, df_labels, col_names = get_preds_targets(images_dirs=None,
                                                        images_paths=images_paths,
                                                        pred_results_dir=pred_results_dir,
                                                        detector=model,
                                                        load_results=args.hn_load_results,
                                                        save_tag="hn-sampling"
                                                    )
    # compute performance & uncertainty of model
    df_results_per_img = compute_detector_performance(df_results,df_labels,col_names)
    df_results_per_img = get_uncertainty(df_results_per_img=df_results_per_img,mode=args.hn_uncertainty_method)
    
    ### save hard samples. Those with low mAP and high or low confidence score
    score_col="max_scores"
    mask_low_map = (df_results_per_img['map50']<args.hn_map_thrs) * (df_results_per_img['map75']<args.hn_map_thrs)
    mask_high_scores = df_results_per_img[score_col]>args.hn_score_thrs
    mask_low_scores =  df_results_per_img[score_col]< (1-args.hn_score_thrs)
    mask_selected = mask_low_map * mask_high_scores + mask_low_map * mask_low_scores + (df_results_per_img['uncertainty']>args.hn_uncertainty_thrs)
    df_hard_negatives = df_results_per_img.loc[mask_selected]

    # save image paths in data_config yaml
    df_hard_negatives['image_paths'].to_csv(save_path_samples,index=False,header=False)
    cfg_dict = {    'path':  data_config_root,
                    'names': yolo_config['names'],
                    'train': os.path.relpath(save_path_samples, start=data_config_root),
                    'val':   [os.path.relpath(os.path.join(yolo_config['path'],p), start=data_config_root) for p in yolo_config['val']],
                    'nc':    yolo_config['nc'],
                }
    with open(save_data_config_yaml,'w') as file:
        yaml.dump(cfg_dict,file)
    
    return str(save_data_config_yaml)
    

def training_routine(model:YOLO,args:Arguments,imgsz:int=None,batchsize:int=None,data_cfg:str|None=None,resume:bool=False):

    # Train the model
    model.train(data=data_cfg or args.data_config_yaml,
                epochs=args.epochs,
                imgsz=imgsz or min(args.height,args.width),
                device=args.device,
                freeze=args.freeze,
                name=args.run_name,
                single_cls=args.is_detector,
                lr0=args.lr0,
                lrf=args.lrf,
                momentum=args.optimizer_momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                dropout=args.dropout,
                batch=batchsize or args.batchsize,
                val=True,
                plots=True,
                cos_lr=args.cos_annealing,
                deterministic=False,
                cache=False, # saves images as *.npy
                optimizer=args.optimizer,
                project=args.project_name,
                patience=args.patience,
                multi_scale=False,
                degrees=args.rotation_degree,
                mixup=args.mixup,
                scale=args.scale,
                mosaic=args.mosaic,
                augment=False,
                erasing=args.erasing,
                copy_paste=args.copy_paste,
                shear=args.shear,
                fliplr=args.fliplr,
                flipud=args.flipud,
                perspective=0.,
                hsv_s=args.hsv_s,
                hsv_h=args.hsv_h,
                hsv_v=args.hsv_v,
                translate=args.translate,
                auto_augment='augmix',
                exist_ok=True,
                seed=args.seed,
                resume=resume
            )


def remove_label_cache(data_config_yaml:str):

     # Remove labels.cache
    try:
        with open(data_config_yaml,'r') as file:
            yolo_config = yaml.load(file,Loader=yaml.FullLoader)
        root = yolo_config["path"]
        for p in yolo_config["train"] + yolo_config["val"]:
            path = os.path.join(root,p,"../labels.cache")
            if os.path.exists(path):
                os.remove(path)
                print(f"Removing: {os.path.join(root,p,"../labels.cache")}")
    except Exception as e:
        # print(e)
        traceback.print_exc()


def pretraining_run(model:YOLO, args:Arguments):

    # check arguments
    assert os.path.exists(args.ptr_data_config_yaml), "provide --ptr-data-config-yaml"
    print("\n\n------------ Pretraining ----------",end="\n\n")
    # remove cache
    remove_label_cache(args.ptr_data_config_yaml)

    # set parameters
    args.epochs = args.ptr_epochs
    args.lr0 = args.ptr_lr0
    args.lrf = args.ptr_lrf
    args.freeze = args.ptr_freeze
    training_routine(model=model,
                        args=args,
                        imgsz=args.ptr_tilesize,
                        batchsize=args.ptr_batchsize,
                        data_cfg=args.ptr_data_config_yaml,
                        resume=False
                )


def hard_negative_strategy_run(model:YOLO, args:Arguments,img_glob_pattern:str="*"):

    # check  arguments
    assert args.hn_save_dir is not None, "Provide --hn-save-dir"
    print("\n\n------------ hard negative sampling learning strategy ----------",end="\n\n")
    # remove cache
    remove_label_cache(args.hn_data_config_yaml)
        
    cfg_path = get_data_cfg_paths_for_cl(ratio=args.hn_ratio,
                                            data_config_yaml=args.hn_data_config_yaml,
                                            cl_save_dir=args.hn_save_dir,
                                            seed=args.seed,
                                            split='train',
                                            pattern_glob=img_glob_pattern
                                        )
    hn_cfg_path = get_data_cfg_paths_for_HN(args=args,
                                                data_config_yaml=cfg_path
                                            )
    args.lr0 = args.hn_lr0
    args.lrf = args.hn_lrf
    args.freeze = args.hn_freeze
    args.epochs = args.hn_num_epochs
    resume = False #args.use_pretraining or args.use_continual_learning
    training_routine(model=model,
                        args=args,
                        imgsz=args.hn_imgsz,
                        batchsize=args.hn_batch_size,
                        data_cfg=hn_cfg_path,
                        resume=resume
                    )


def continual_learning_run(model:YOLO,args:Arguments,img_glob_pattern:str="*"):

    # check arguments
    assert os.path.exists(args.cl_data_config_yaml), "Provide --cl-data-config-yaml"
    print("\n\n------------ Continual learning ----------",end="\n\n")
    # remove cache
    remove_label_cache(args.cl_data_config_yaml)
    # check flags
    for flag in (args.cl_ratios,args.cl_epochs,args.cl_freeze):
        assert len(flag) == len(args.cl_lr0s), "all args.cl_* flags should have the same length."
    
    # get yaml data_cfg files for CL runs
    count = 0
    for lr, ratio, num_epochs,freeze in zip(args.cl_lr0s,args.cl_ratios,args.cl_epochs,args.cl_freeze):
        cl_cfg_path = get_data_cfg_paths_for_cl(ratio=ratio,
                                                data_config_yaml=args.cl_data_config_yaml,
                                                cl_save_dir=args.cl_save_dir,
                                                seed=args.seed,
                                                split='train',
                                                pattern_glob=img_glob_pattern
                                            )
        # freeze layer. see ultralytics docs :)
        args.freeze = freeze
        args.lr0 = lr
        args.epochs = num_epochs
        resume = False #args.use_pretraining or (count>0)
        training_routine(model=model,
                        args=args,
                        data_cfg=cl_cfg_path,
                        batchsize=args.cl_batch_size,
                        resume=resume
                    )
        count += 1


def start_training(args:Arguments):
    """Trains a YOLO model using ultralytics.

    Args:
        args (Arguments): configs
    """

    # logger = logging.getLogger(__name__)

    # Load a pre-trained model
    model = YOLO(args.path_weights,task='detect',verbose=False)

    # Display model information (optional)
    model.info()    

    # pretraining
    if args.use_pretraining:
        pretraining_run(model=model,args=args)

    # Continual learning strategy    
    if args.use_continual_learning:
        continual_learning_run(model=model, args=args)

    # hard negative sampling learning strategy
    if args.use_hn_learning:
        hard_negative_strategy_run(model=model,args=args)

    # standard training routine
    if not (args.ptr_data_config_yaml or args.use_continual_learning or args.use_hn_learning):
        training_routine(model=model,args=args)

    





