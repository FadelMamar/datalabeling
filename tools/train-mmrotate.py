# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:30:13 2025

@author: FADELCO
"""

import mmcv
import yaml
import os
import os.path as osp
import copy
import numpy as np
from pathlib import Path
import pandas as pd
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
import torch
from mmdet.apis import set_random_seed
from mmrotate.core import poly2obb_np
from typing import Sequence
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from datargs import parse
from dataclasses import dataclass
from mmrotate.utils import (collect_env, get_root_logger,
                            setup_multi_processes)

def load_yaml(data_config_yaml: str) -> dict:
    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    return data_config


@dataclass
class Flags:
    # main config
    config: str = None  # e.g. oriented_rcnn_r50_fpn_1x_dota_le90.py

    # data_config file
    data_config: str = None

    # inference
    slice_size: int = 800
    overlap_ratio: float = 0.2
    combine_method: str = "nms"
    match_threshold: float = 0.6
    match_metric: str = "ios"
    draw_threshold: float = 0.5
    save_results: bool = False
    slice_infer: bool = False

    save_threshold: float = 0.5
    rtn_im_file: bool = False

    weights: str = None

    infer_dir: str = None
    infer_img: str = None
    infer_list: str = None

    # training
    optimizer: str = "SGD"  # SGD, Adam
    batch_size: int = 16
    resume: bool = False
    lr0: float = 1e-3
    epoch: int = 30
    empty_ratio: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 41
    num_workers: int = 8
    logging_interval: int = 100
    freeze_ratio: float = 0.0

    # evaluation
    eval_interval: int = 1
    checkpoint_interval: int = 1
    enable_val:bool=False

    # logging
    output_dir: str = "runs-mmrotate"
    mlflow_tracking_uri: str = "http://localhost:5000"
    project_name: str = "wildAI-detection"
    run_name: str = "run"
    use_wandb: bool = False
    wandb_entity:str ="ipeo-epfl"
    tags: Sequence[str] = None


# set the empty ratio to 0.0
os.environ["EMPTY_RATIO"] = str(0.0)


@ROTATED_DATASETS.register_module()
class WildAIDataset(DOTADataset):
    """Dataset for detection."""

    def __init__(self, ann_file, pipeline, version="oc", difficulty=100, **kwargs):
        self.version = version
        self.difficulty = difficulty
        self.empty_ratio = max(float(os.environ["EMPTY_RATIO"]), 0.0)

        super(WildAIDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_folder):
        """
        Args:
            ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {
            c: i for i, c in enumerate(self.CLASSES)
        }  # in mmdet v2.0 label is 0-based
        # ann_files = Path(ann_folder).glob('*.txt')
        image_paths = list(
            (Path(ann_folder).parent / "images").glob("*")
        )  # negative and positive samples
        positive_images = [
            img
            for img in image_paths
            if Path(str(img).replace("images", "dota_labels"))
            .with_suffix(".txt")
            .exists()
        ]
        ann_files = [
            Path(str(img).replace("images", "dota_labels")).with_suffix(".txt")
            for img in positive_images
        ]

        negative_images = list(set(image_paths) - set(positive_images))

        num_sampled_empty = int(
            min(self.empty_ratio * len(ann_files), len(negative_images))
        )
        negative_images = (
            pd.Series(negative_images).sample(num_sampled_empty, frac=None).to_list()
        )

        selected_images_paths = negative_images + positive_images
        ann_files = [None] * len(negative_images) + ann_files

        # print(len(selected_images_paths))

        data_infos = []
        if not ann_files:  # test phase
            for img in selected_images_paths:
                data_info = {}
                data_info["filename"] = img.name
                data_info["ann"] = {}
                data_info["ann"]["bboxes"] = []
                data_info["ann"]["labels"] = []
                data_infos.append(data_info)
        else:
            for image_path, ann_file in zip(selected_images_paths, ann_files):
                data_info = {}
                data_info["filename"] = image_path.name
                data_info["ann"] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if ann_file is not None:
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
                    data_info["ann"]["bboxes"] = np.array(gt_bboxes, dtype=np.float32)
                    data_info["ann"]["labels"] = np.array(gt_labels, dtype=np.int64)
                    data_info["ann"]["polygons"] = np.array(
                        gt_polygons, dtype=np.float32
                    )
                else:
                    data_info["ann"]["bboxes"] = np.zeros((0, 5), dtype=np.float32)
                    data_info["ann"]["labels"] = np.array([], dtype=np.int64)
                    data_info["ann"]["polygons"] = np.zeros((0, 8), dtype=np.float32)

                if gt_polygons_ignore:
                    data_info["ann"]["bboxes_ignore"] = np.array(
                        gt_bboxes_ignore, dtype=np.float32
                    )
                    data_info["ann"]["labels_ignore"] = np.array(
                        gt_labels_ignore, dtype=np.int64
                    )
                    data_info["ann"]["polygons_ignore"] = np.array(
                        gt_polygons_ignore, dtype=np.float32
                    )
                else:
                    data_info["ann"]["bboxes_ignore"] = np.zeros(
                        (0, 5), dtype=np.float32
                    )
                    data_info["ann"]["labels_ignore"] = np.array([], dtype=np.int64)
                    data_info["ann"]["polygons_ignore"] = np.zeros(
                        (0, 8), dtype=np.float32
                    )

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x["filename"][:-4], data_infos)]
        return data_infos


if __name__ == "__main__":
    from datargs import parse
    import mlflow
    import time

    args = parse(Flags)

    # checks
    assert 0 <= args.freeze_ratio <= 1, "freeze_ratio should be in [0, 1]"

    # update run_name
    args.run_name = f"{args.run_name}#empty_{args.empty_ratio}#freeze_{args.freeze_ratio}"

    # update output_dir
    args.output_dir = osp.join(args.output_dir, args.run_name)

    # set mlflow uri
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # set the empty ratio of the dataset: the quantity of empty images
    # to be sampled from the negative samples
    os.environ["EMPTY_RATIO"] = str(args.empty_ratio)

    # get info on the dataset
    data_config = load_yaml(args.data_config)
    root_dir = data_config["path"]
    num_classes = data_config["nc"]
    classes = [
        data_config["names"][i] for i in range(num_classes)
    ]
    
    # Load the config
    cfg = mmcv.Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    dataset_type = "WildAIDataset"
    # Modify dataset type and path
    cfg.dataset_type = dataset_type
    cfg.data_root = root_dir
    cfg.classes = classes

    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.num_workers

    cfg.data.test.type = dataset_type
    cfg.data.test.data_root = root_dir
    cfg.data.test.img_prefix = data_config["test"][0]
    cfg.data.test.ann_file = str(data_config["test"][0]).replace(
        "images", "dota_labels"
    )
    cfg.data.test.classes = classes

    cfg.data.train.type = dataset_type
    cfg.data.train.data_root = root_dir
    cfg.data.train.img_prefix = data_config["train"][0]
    cfg.data.train.ann_file = str(data_config["train"][0]).replace(
        "images", "dota_labels"
    )
    cfg.data.train.classes = classes

    cfg.data.val.type = dataset_type
    cfg.data.val.data_root = root_dir
    cfg.data.val.img_prefix = data_config["val"][0]
    cfg.data.val.ann_file = str(data_config["val"][0]).replace("images", "dota_labels")
    cfg.data.val.classes = classes

    # modify num classes of the model in box head
    cfg.num_classes = num_classes
    try:
        cfg.model.roi_head.bbox_head.num_classes = num_classes
    except Exception as e:
        if isinstance(cfg.model.roi_head.bbox_head, list):
            for head in cfg.model.roi_head.bbox_head:
                head.num_classes = num_classes
        else:
            print(e)
            raise NotImplementedError("Head type not implemented")

    # use the mask branch
    cfg.load_from = args.weights

    # Set up working dir to save files and logs.
    cfg.work_dir = args.output_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # optimizer
    if args.optimizer == "SGD":
        cfg.optimizer = dict(type="SGD", lr=args.lr0, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == "Adam":
        cfg.optimizer = dict(
            type="Adam", lr=args.lr0, betas=(0.9, 0.999), weight_decay=1e-4
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    cfg.lr_config.warmup = 'linear'
    cfg.runner.max_epochs = args.epoch
    cfg.total_epochs = args.epoch
    cfg.log_config.interval = args.logging_interval

    # Change the evaluation settings
    cfg.evaluation.metric = "mAP"
    cfg.evaluation.interval = args.eval_interval
    cfg.evaluation.save_best = "mAP"
    cfg.evaluation.rule = "greater"
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = args.checkpoint_interval

    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set seed thus the results are more reproducible
    cfg.seed = args.seed
    set_random_seed(cfg.seed, deterministic=False)
    meta['seed'] = cfg.seed
    meta['exp_name'] = osp.basename(args.run_name)

    # set computing device
    cfg.gpu_ids = (
        range(1) if torch.cuda.device_count() < 2 else range(torch.cuda.device_count())
    )
    cfg.device = args.device

    # log the training process
    # ref: https://mmcv.readthedocs.io/en/v1.3.6/_modules/mmcv/runner/hooks/logger/wandb.html
    cfg.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(type="MlflowLoggerHook",exp_name=args.project_name,by_epoch=True,log_model=True,tags=dict(name=args.run_name)
        )
    ]
    if args.use_wandb:
        cfg.log_config.hooks.append(dict(type="WandbLoggerHook", init_kwargs={"project": args.project_name,
                                                                              "name": args.run_name,
                                                                              "tags": args.tags, 
                                                                              "entity": args.wandb_entity
                                                                            }       
                                        )                              
                                )
    # set workflow
    if args.enable_val:
        cfg.workflow = [('train', 1), ('val', 1)]
    else:
        cfg.workflow = [('train', 1)]

    # build dataset
    datasets = [build_dataset(cfg.data.train)]
    # a bit ambiguous --> see https://github.com/open-mmlab/mmrotate/blob/main/tools/train.py#L170
    # explanation regarding workflow @ https://mmrotate.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    print(datasets)

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    logger.info(f"\nNumber of parameters: {sum([torch.numel(p) for p in model.parameters()])/1e6:.2f} M")

    # Freeze params
    if args.freeze_ratio:
        num_layers = len(list(model.parameters()))
        for idx, param in enumerate(model.parameters()):
            if idx / num_layers < args.freeze_ratio:
                param.requires_grad = False
            else:
                break
        print(f"{int(num_layers * args.freeze_ratio)}/{num_layers} layers have been frozen.\n")

    # Create work_dir
    train_detector(model, datasets, cfg, distributed=False, validate=args.enable_val)
