"""To train using paddlepaddle, do the following:
    - Create a virtual environment Python 3.10.16
    - Follow the steps at this page : https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html & https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/tutorials/INSTALL.md
    - Prepare data : https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/tutorials/data/PrepareDetDataSet_en.md
    - Customize yaml files for training: https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/advanced_tutorials/customization/detection_en.md &
                https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation_en.md
    - Download the pretrained weights inside one of the yaml profiles

Returns:
    _type_: _description_
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from dataclasses import dataclass
from pathlib import Path
# ignore warning log
import warnings
warnings.filterwarnings("ignore")
import glob
import ast
import json
import yaml
from typing import Sequence
import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer, Trainer_ARSL
from ppdet.utils.logger import setup_logger

def load_yaml(data_config_yaml: str) -> dict:
    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    return data_config


@dataclass
class Flags:
    # main ppd yaml config
    config: str = None
    data_config:str = None

    # inference
    slice_size: int = 800
    overlap_ratio: float = 0.2
    combine_method: str = "nms"
    match_threshold: float = 0.6
    match_metric: str = "ios"
    draw_threshold: float = 0.5
    save_results: bool = False
    slice_infer: bool = False

    visualize: ast.literal_eval = False
    save_threshold: float = 0.5
    do_eval: ast.literal_eval = False
    rtn_im_file: bool = False

    weights: str = None

    infer_dir: str = None
    infer_img: str = None
    infer_list: str = None

    tr_empty_ratio: float = 0.1
    val_empty_ratio: float = 1.0

    freeze_ratio: float= 0.

    # training
    resume: bool = False
    amp: bool = False
    print_flops: bool = False
    print_params: bool = False
    lr0: float = 1e-3
    epoch: int = 30
    device: str = "cuda"
    tr_batchsize: int = 8
    val_batchsize:int = 16

    # evaluation
    eval_interval:int=2

    # logging
    output_dir: str = "runs-ppd"
    mlflow_tracking_uri: str = "http://localhost:5000"
    project_name: str = "wildAI-detection"
    run_name: str = "run-ppd"
    use_wandb: bool = False
    tags: Sequence[str] = None
    log_iter:int=100 # log every 100 iterations


def train_ppd(args: Flags):
    logger = setup_logger("train")

    data_config = load_yaml(args.data_config)
    root_dir = os.path.join(data_config["path"], "coco-dataset")

    cfg = load_config(args.config)

    args.run_name = args.run_name + f"-tr_empty_ratio_{args.tr_empty_ratio}_freeze_{args.freeze_ratio}"
    args.output_dir = os.path.join("runs_ppd", args.run_name)

    cfg["num_classes"] = data_config["nc"]
    cfg["amp"] = args.amp
    cfg["save_dir"] = args.output_dir
    cfg["print_flops"] = args.print_flops
    cfg["print_params"] = args.print_params
    cfg["epoch"] = args.epoch
    cfg["log_iter"] = args.log_iter
    cfg["use_wandb"] = args.use_wandb
    cfg["LearningRate"]["base_lr"] = args.lr0
    cfg["LearningRate"]["schedulers"] =  [ {"name": "CosineDecay",
                                            "max_epochs": args.epoch
                                            },
                                            {"name": "LinearWarmup", 
                                            "start_factor": 0.0, 
                                            "epochs": 1
                                            }
                                        ]
    cfg["wandb"] = {
        "project": args.project_name,
        "name": args.run_name,
        "tags": args.tags,
        "entity": "ipeo-epfl",
    }
    cfg['use_gpu'] = (args.device == "cuda")

    cfg['weights'] = os.path.join(args.output_dir,'model_final')
    cfg["pretrain_weights"] = args.weights

    # set batchsize
    cfg["TrainReader"]["batch_size"] = args.tr_batchsize
    cfg["EvalReader"]["batch_size"] = args.val_batchsize
    cfg["TestReader"]["batch_size"] = args.val_batchsize

    # set datasets
    cfg["TrainDataset"]["dataset_dir"] = root_dir
    cfg["TrainDataset"]["image_dir"] = "train"
    cfg["TrainDataset"]["anno_path"] = "annotations/annotations_train.json"
    cfg["TrainDataset"]["empty_ratio"] = args.tr_empty_ratio
    cfg["TrainDataset"]["allow_empty"] = args.tr_empty_ratio>0.

    cfg["EvalDataset"]["dataset_dir"] = root_dir
    cfg["EvalDataset"]["image_dir"] = "val"
    cfg["EvalDataset"]["anno_path"] = "annotations/annotations_val.json"
    cfg["EvalDataset"]["empty_ratio"] = args.val_empty_ratio
    cfg["EvalDataset"]["allow_empty"] = args.val_empty_ratio>0.

    cfg["TestDataset"]["dataset_dir"] = root_dir
    cfg["TestDataset"]["image_dir"] = "test"
    cfg["TestDataset"]["empty_ratio"] = 1.0
    cfg["TestDataset"]["anno_path"] = "annotations/annotations_test.json"
    cfg["TestDataset"]["allow_empty"] = True

    # eval metrics computation interval
    cfg["snapshot_epoch"] = args.eval_interval

    if cfg.use_gpu:
        place = paddle.set_device("gpu")
    elif cfg.use_npu:
        place = paddle.set_device("npu")
    elif cfg.use_xpu:
        place = paddle.set_device("xpu")
    elif cfg.use_mlu:
        place = paddle.set_device("mlu")
    else:
        place = paddle.set_device("cpu")
    
    logger.info(f"Using device: {place}")

    trainer = Trainer(cfg, mode="train")

    if args.resume:
        trainer.resume_weights(args.resume)
    elif "pretrain_weights" in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)
    else:
        logger.info("No weights loaded.")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(json.dumps(cfg, indent=4))

    
    # disable grad for some parameters
    num_layers = len(list(trainer.model.parameters()))
    for idx, param in enumerate(trainer.model.parameters()):
        if idx / num_layers < args.freeze_ratio:
            param.trainable = False
        else:
            break
    logger.info(f"\n{int(num_layers * args.freeze_ratio)}/{num_layers} layers have been frozen.\n")

    trainer.train(args.do_eval)


def get_test_images(infer_dir: str, infer_img: str, infer_list=None):
    """
    Get image path list in TEST mode
    """
    logger = setup_logger("test")
    assert infer_img is not None or infer_dir is not None, (
        "--infer_img or --infer_dir should be set"
    )
    assert infer_img is None or os.path.isfile(infer_img), "{} is not a file".format(
        infer_img
    )
    assert infer_dir is None or os.path.isdir(infer_dir), (
        "{} is not a directory".format(infer_dir)
    )

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), "infer_dir {} is not a directory".format(infer_dir)
    if infer_list:
        assert os.path.isfile(infer_list), (
            f"infer_list {infer_list} is not a valid file path."
        )
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            images.update([os.path.join(infer_dir, line.strip())])
    else:
        exts = ["jpg", "jpeg", "png", "bmp"]
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob("{}/*.{}".format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def inference(images, args: Flags):
    cfg = load_config(args.config)

    if args.rtn_im_file:
        cfg["TestReader"]["sample_transforms"][0]["Decode"]["rtn_im_file"] = (
            args.rtn_im_file
        )
    ssod_method = cfg.get("ssod_method", None)
    if ssod_method == "ARSL":
        trainer = Trainer_ARSL(cfg, mode="test")
        trainer.load_weights(args.weights, ARSL_eval=True)
    else:
        trainer = Trainer(cfg, mode="test")
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
            visualize=args.visualize,
        )
    else:
        trainer.predict(
            images,
            draw_threshold=args.draw_threshold,
            output_dir=args.output_dir,
            save_results=args.save_results,
            visualize=args.visualize,
            save_threshold=args.save_threshold,
            do_eval=args.do_eval,
        )


if __name__ == "__main__":
    from datargs import parse

    args = parse(Flags)

    train_ppd(args)
