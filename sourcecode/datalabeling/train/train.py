from ultralytics import YOLO
import yaml
from ..arguments import Arguments
import os, logging, traceback
from pathlib import Path
import pandas as pd
import math


def sample_pos_neg(images_paths:list,ratio:float,seed:int=41):

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


def get_data_cfg_paths_for_cl(ratio:float,data_config_yaml:str,cl_save_dir:str,seed:int=41,split:str='train'):

    with open(data_config_yaml,'r') as file:
        yolo_config = yaml.load(file,Loader=yaml.FullLoader)

    root = yolo_config["path"]
    train_dirs_images = [os.path.join(root,p) for p in yolo_config[split]]
    
    # sample positive and negative images
    sampled_imgs_paths = []
    for dir_images in train_dirs_images:
        print(f"Sampling positive and negative samples from {dir_images}")
        paths = sample_pos_neg(images_paths=list(Path(dir_images).iterdir()),
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
    save_path_cfg = Path(save_path_samples).with_suffix('.yml')
    with open(save_path_cfg,'w') as file:
        yaml.dump(cfg_dict,file)

    print(f"Saving samples at: {save_path_samples} and data_cfg at {save_path_cfg}",end="\n\n")

    return str(save_path_cfg)


def training_routine(model:YOLO,args:Arguments,data_cfg:str|None=None,num_epochs:int=None,resume:bool=False):

    # Train the model
    model.train(data=data_cfg or args.data_config_yaml,
                epochs=num_epochs or args.epochs,
                imgsz=min(args.height,args.width),
                device=args.device,
                freeze=args.freeze,
                name=args.run_name,
                single_cls=args.is_detector,
                lr0=args.lr0,
                lrf=args.lrf,
                momentum=args.optimizer_momentum,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                batch=args.batchsize,
                val=True,
                plots=True,
                cos_lr=args.cos_annealing,
                deterministic=False,
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


def start_training(args:Arguments):
    """Trains a YOLO model using ultralytics. By defaults, it will compile new 'labels.cache' files.

    Args:
        args (Arguments): configs
    """

    logger = logging.getLogger(__name__)

    # Load a pre-trained model
    model = YOLO(args.path_weights,task='detect',verbose=False)

    # Display model information (optional)
    model.info()    

    # Remove labels.cache
    try:
        with open(args.data_config_yaml,'r') as file:
            yolo_config = yaml.load(file,Loader=yaml.FullLoader)
        root = yolo_config["path"]
        for p in yolo_config["train"] + yolo_config["val"]:
            path = os.path.join(root,p,"../labels.cache")
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removing: {os.path.join(root,p,"../labels.cache")}")
    except Exception as e:
        # print(e)
        traceback.print_exc()

    # Continual learning strategy

    # check arguments
    if args.use_continual_learning:
        for flag in (args.cl_ratios,args.cl_epochs,args.cl_freeze):
            assert len(flag) == len(args.cl_lr0s), "all args.cl_* flags should have the same length."

        # get data_cfg files for CL runs
        for lr, ratio, num_epochs,freeze in zip(args.cl_lr0s,args.cl_ratios,args.cl_epochs,args.cl_freeze):
            cl_cfg_path = get_data_cfg_paths_for_cl(ratio=ratio,
                                                    data_config_yaml=args.data_config_yaml,
                                                    cl_save_dir=args.cl_save_dir,
                                                    seed=args.seed,
                                                    split='train'
                                                )
            # freeze layer. see ultralytics docs :)
            args.freeze = freeze
            args.lr0 = lr
            training_routine(model=model,
                             args=args,
                             data_cfg=cl_cfg_path,
                             num_epochs=num_epochs
                            )
    
    else:
        training_routine(model=model,args=args)

    





