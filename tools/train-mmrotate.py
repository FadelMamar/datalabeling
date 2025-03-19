# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:30:13 2025

@author: FADELCO
"""
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector
import yaml
import glob
import os
import os.path as osp
import numpy as np
from pathlib import Path
import pandas as pd
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
import glob
import os
import os.path as osp
import re
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from functools import partial
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
import os.path as osp
from typing import Sequence
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from datargs import parse
from dataclasses import dataclass

def load_yaml(data_config_yaml: str)->dict:
    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    
    return data_config


@dataclass
class Flags:

    # main config
    config:str= None # e.g. oriented_rcnn_r50_fpn_1x_dota_le90.py

    # data_config file
    data_config:str=None

    # inference
    slice_size:int=800
    overlap_ratio:float=0.2
    combine_method:str='nms'
    match_threshold:float=0.6
    match_metric:str='ios'
    draw_threshold:float=0.5
    save_results:bool=False
    slice_infer:bool=False

    save_threshold:float=0.5
    rtn_im_file:bool=False

    weights:str=None

    infer_dir:str=None
    infer_img:str=None
    infer_list:str=None

    # training
    optimizer:str='Adam' # SGD, Adam
    batch_size:int=16
    resume:bool=False
    lr0:float=1e-3
    epoch:int=30
    empty_ratio:float=0.0
    device:str='cuda' if torch.cuda.is_available() else 'cpu'
    seed:int=41
    num_workers:int=4

    # evaluation
    eval_interval:int=3
    checkpoint_interval:int=3

    # logging
    output_dir:str="runs-mmrotate"
    mlflow_tracking_uri: str = "http://localhost:5000"
    project_name: str = "wildAI-detection"
    run_name:str = "run"
    use_wandb:bool=False
    tags:Sequence[str]=None

# set the empty ratio to 0.0
os.environ["EMPTY_RATIO"]=str(0.0)

@ROTATED_DATASETS.register_module()
class WildAIDataset(DOTADataset):
    """Dataset for detection."""

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty
        self.empty_ratio = max(float(os.environ["EMPTY_RATIO"]), 0.)

        super(WildAIDataset, self).__init__(ann_file, pipeline, **kwargs)
    
    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        # ann_files = Path(ann_folder).glob('*.txt')
        image_paths = list((Path(ann_folder).parent/'images').glob('*')) # negative and positive samples
        positive_images = [img for img in image_paths if Path(str(img).replace('images','dota_labels')).with_suffix('.txt').exists()]
        ann_files = [Path(str(img).replace('images','dota_labels')).with_suffix('.txt') for img in positive_images]

        negative_images = list(set(image_paths) - set(positive_images))

        num_sampled_empty = int(min(self.empty_ratio*len(ann_files), len(negative_images)))
        negative_images = pd.Series(negative_images).sample(num_sampled_empty,frac=None).to_list()

        selected_images_paths = negative_images + positive_images
        ann_files = [None]*len(negative_images)   +  ann_files

        # print(len(selected_images_paths))

        data_infos = []
        if not ann_files:  # test phase
            for img in selected_images_paths:
                data_info = {}
                data_info['filename'] = img.name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for image_path,ann_file in zip(selected_images_paths,ann_files):
                data_info = {}
                data_info['filename'] = image_path.name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if ann_file is not None :
                    with open(ann_file) as f:
                        s = f.readlines()
                        for si in s:
                            bbox_info = si.split()
                            poly = np.array(bbox_info[:8], dtype=np.float32)
                            try:
                                x, y, w, h, a = poly2obb_np(poly, self.version)
                            except Exception as e:  # noqa: E722
                                print(e)
                                continue
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[9])
                            label = cls_map[cls_name]
                            if difficulty > self.difficulty:
                                pass
                            else:
                                gt_bboxes.append([x, y, w, h, a])
                                gt_labels.append(label)
                                gt_polygons.append(poly)
                
                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos
    
if __name__ == "__main__":
    from datargs import parse
    import mlflow

    args = parse(Flags)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.project_name)

    os.environ["EMPTY_RATIO"] = str(args.empty_ratio)

    data_config = load_yaml(args.data_config)
    root_dir = data_config["path"]
    num_classes = data_config["nc"]
    classes = [data_config["names"][i] for i in range(num_classes)] # e.g. ['sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6']
    
    # Load the config
    cfg = mmcv.Config.fromfile(args.config)

    dataset_type = 'WildAIDataset'
    # Modify dataset type and path
    cfg.dataset_type = dataset_type
    cfg.data_root = root_dir

    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.num_workers

    cfg.data.test.type = dataset_type
    cfg.data.test.data_root = root_dir
    cfg.data.test.img_prefix = data_config["test"][0]
    cfg.data.test.ann_file = str(data_config["test"][0]).replace('images','dota_labels')
    cfg.data.test.classes = classes

    cfg.data.train.type = dataset_type
    cfg.data.train.data_root = root_dir
    cfg.data.train.img_prefix = data_config["train"][0]
    cfg.data.train.ann_file =  str(data_config["train"][0]).replace('images','dota_labels')
    cfg.data.train.classes = classes

    cfg.data.val.type = dataset_type
    cfg.data.val.data_root = root_dir
    cfg.data.val.img_prefix = data_config["val"][0]
    cfg.data.val.ann_file = str(data_config["val"][0]).replace('images','dota_labels')
    cfg.data.val.classes = classes

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = num_classes

    # use the mask branch
    cfg.load_from = args.weights

    # Set up working dir to save files and logs.
    cfg.work_dir = args.output_dir

    # optimizer
    if args.optimizer == 'SGD':
        cfg.optimizer = dict(type='SGD', lr=args.lr0, momentum=0.9, weight_decay=1e-4)
    else:
        cfg.optimizer = dict(type='Adam', lr=args.lr0, betas=(0.9,0.999), weight_decay=1e-4)
    cfg.lr_config.warmup = None
    cfg.runner.max_epochs = args.epoch
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = args.eval_interval
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = args.checkpoint_interval

    # Set seed thus the results are more reproducible
    cfg.seed = args.seed
    set_random_seed(cfg.seed, deterministic=False)
    cfg.gpu_ids = range(1) if torch.cuda.device_count()<2 else torch.cuda.device_count()
    cfg.device= args.device

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    datasets = [build_dataset(cfg.data.train)]

    print(datasets)

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Load checkpoint
    model = model.to(cfg.device)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)