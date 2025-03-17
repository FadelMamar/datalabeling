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
        trainer.train(args.do_eval)


def get_test_images(infer_dir:str, infer_img:str, infer_list=None):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    if infer_list:
        assert os.path.isfile(
            infer_list), f"infer_list {infer_list} is not a valid file path."
        with open(infer_list, 'r') as f:
            lines = f.readlines()
        for line in lines:
            images.update([os.path.join(infer_dir, line.strip())])
    else:
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def inference(images,args:Flags):

    cfg = load_config(args.config)

    if args.rtn_im_file:
        cfg['TestReader']['sample_transforms'][0]['Decode'][
            'rtn_im_file'] = args.rtn_im_file
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        trainer = Trainer_ARSL(cfg, mode='test')
        trainer.load_weights(args.weights, ARSL_eval=True)
    else:
        trainer = Trainer(cfg, mode='test')
        trainer.load_weights(args.weights)
    
    if args.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=args.slice_size,
            overlap_ratio=args.overlap_ratio,
            combine_method=args.combine_method,
            match_threshold=args.match_threshold,
            match_metric=args.match_metric,
            draw_threshold=args.draw_threshold,
            output_dir=args.output_dir,
            save_results=args.save_results,
            visualize=args.visualize)
    else:
        trainer.predict(
            images,
            draw_threshold=args.draw_threshold,
            output_dir=args.output_dir,
            save_results=args.save_results,
            visualize=args.visualize,
            save_threshold=args.save_threshold,
            do_eval=args.do_eval)
