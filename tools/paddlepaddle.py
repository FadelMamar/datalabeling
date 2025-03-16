from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from dataclasses import dataclass

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast
import mlflow

import paddle
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_gcu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

@dataclass
class Flags:
    config:str= None

    # inference
    slice_size:int=640
    overlap_ratio:float=0.25
    combine_method:str='nms'
    match_threshold:float=0.6
    match_metric:str='ios'
    draw_threshold:float=0.5
    save_results:bool=True
    slice_infer:bool=False

    visualize:ast.literal_eval=True
    save_threshold:float=0.5
    do_eval:ast.literal_eval=False
    rtn_im_file:bool=False

    weights:str=None

    infer_dir:str=None
    infer_img:str=None
    infer_list:str=None

    # training
    resume:bool=False
    eval:bool=False

    # logging
    output_dir:str=None
    mlflow_tracking_uri: str = "http://localhost:5000"
    project_name: str = "wildAI-detection"
    run_name:str = "run-ppd"


def train_ppd(args:Flags):

    cfg = load_config(args.config)

    trainer = Trainer(cfg, mode='train')

    if args.resume:
        trainer.resume_weights(args.resume)
    trainer.load_weights(args.weights)

    # training
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.project_name)
    mlflow.paddle.autolog(log_every_n_epoch = 5)
    with mlflow.start_run(run_name=args.run_name) as run:
        trainer.train(flags.eval)
