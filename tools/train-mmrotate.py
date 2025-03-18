# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:30:13 2025

@author: FADELCO
"""
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector

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

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


EMPTY_RATIO=0.1
@ROTATED_DATASETS.register_module()
class TinyDataset(DOTADataset):
    """SAR ship dataset for detection."""

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty
        self.empty_ratio = max(EMPTY_RATIO,0.)

        super(TinyDataset, self).__init__(ann_file, pipeline, **kwargs)
    
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

# Choose to use a config and initialize the detector
config = r'D:\datalabeling\notebooks\oriented_rcnn_r50_fpn_1x_dota_le90.py'
# Setup a checkpoint file to load
checkpoint = r'D:\datalabeling\notebooks\oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

# Set the device to be used for evaluation
# device='cpu'

# Load the config
config = mmcv.Config.fromfile(config)

# Set pretrained to be None since we do not need pretrained model here
# config.model.pretrained = None

# # Initialize the detector
# model = build_detector(config.model)

# # Load checkpoint
# checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# # Set the classes of models for inference
# model.CLASSES = checkpoint['meta']['CLASSES']

# # We need to set the model's cfg for inference
# model.cfg = config

# # Convert the model to GPU
# model.to(device)

# Convert the model into evaluation mode
# model.eval()


dataset_type = 'TinyDataset'
# Modify dataset type and path
cfg.dataset_type = dataset_type
cfg.data_root = r"D:\general_dataset\tiled-data"

cfg.data.test.type = dataset_type
cfg.data.test.data_root = r"D:\general_dataset\tiled-data\test"
cfg.data.test.ann_file = 'dota_labels'
cfg.data.test.img_prefix = 'images/'

cfg.data.train.type = dataset_type
cfg.data.train.data_root = r"D:\general_dataset\tiled-data\val"
cfg.data.train.ann_file =  "dota_labels"
cfg.data.train.img_prefix = "images/"

cfg.data.val.type = dataset_type
cfg.data.val.data_root = r"D:\general_dataset\tiled-data\val"
cfg.data.val.ann_file = 'dota_labels'
cfg.data.val.img_prefix = 'images/'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 6
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

cfg.optimizer.lr = 0.001
cfg.lr_config.warmup = None
cfg.runner.max_epochs = 3
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 3

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device='cpu'

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)