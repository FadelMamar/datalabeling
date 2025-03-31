from datargs import parse
from dataclasses import dataclass
from typing import Sequence
import json
from datalabeling.train.herdnet import HerdnetData, HerdnetTrainer
import lightning as L
import yaml
import torch
from lightning.pytorch.loggers import MLFlowLogger
import animaloc
from animaloc.eval import PointsMetrics, HerdNetEvaluator
from animaloc.vizual import PlotPrecisionRecall
from animaloc.utils.useful_funcs import current_date, mkdir


@dataclass
class Flags:
    
    # data_config file
    data_config: str = None

    # split
    splits:Sequence[str] = ("val",)

    # weights
    weights:str=None

    # inference
    batch_size: int = 16
    imgsz: int = 800
    iou_threshold: float = 0.6
    conf_threshold: float= 0.25
    device:str = "cuda"
    augment:bool=False # Enables test-time augmentation (TTA) during validation, potentially improving detection accuracy at the cost of inference speed
    is_detector:bool=False
    down_ratio:int=2
    eval_radius:int=20

    # logging
    save_csv:bool = False 
    name:str = "val"
    save_dir:str=""
    project_name:str="Herdnet"
    plots:bool=False
    log_mlflow:bool=False

def herdnet_val(args:Flags):
    
    # Get number of classes
    with open(args.data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    num_classes = data_config["nc"] + 1

    # set model
    if args.log_mlflow:
        mlf_logger = MLFlowLogger(
            experiment_name="Herdnet",
            run_name="herdnet-validate",
            tracking_uri=args.mlflow_tracking_uri,
            log_model=True,
        )
    else:
        mlf_logger=None

    herdnet_trainer = HerdnetTrainer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_config_yaml=args.data_config_yaml,
        lr=None,
        weight_decay=None,
        herdnet_model_path=None,
        loaded_weights_num_classes=num_classes,
        ce_weight=None,
        eval_radius=args.eval_radius,
        map_location="cpu",
        strict=True,
        work_dir=args.save_dir,
    )

    # Data
    datamodule = HerdnetData(
        data_config_yaml=args.data_config_yaml,
        patch_size=args.imgsz,
        batch_size=args.batchsize,
        down_ratio=down_ratio,
        train_empty_ratio=0.0,
    )
    
    # Predict
    # images_path = os.path.join(data_config['path'],data_config['test'][0])
    # images_path = list(Path(images_path).glob('*'))
    # datamodule.set_predict_dataset(images_path=images_path,batchsize=1)

    trainer = L.Trainer(accelerator="auto", logger=mlf_logger)

    for split in args.splits:

        if split == 'val':

            datamodule.setup('fit')

            herdnet_evaluator = HerdNetEvaluator(
            model=herdnet_trainer.model,
            dataloader=datamodule.val_dataloader(),
            metrics=herdnet_trainer.metrics_val,
            stitcher=herdnet_trainer.stitcher,
            work_dir=args.save_dir,
            header="validation",
            lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 3.0, "neg_ts": 0.1},
            )
            # out = trainer.validate(model=herdnet_trainer,datamodule=datamodule)

        elif split == 'test':

            datamodule.setup('test')

            herdnet_evaluator = HerdNetEvaluator(
            model=herdnet_trainer.model,
            dataloader=datamodule.test_dataloader(),
            metrics=herdnet_trainer.metrics_val,
            stitcher=herdnet_trainer.stitcher,
            work_dir=args.save_dir,
            header="test",
            lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 3.0, "neg_ts": 0.1},
            )
            # out = trainer.test(model=herdnet_trainer, datamodule=datamodule)

        else:
            raise NotImplementedError
        
        # Run evaludation
        out = herdnet_evaluator.evaluate(wandb_flag=False, viz=False)

        # Save results
        print('Saving the results ...')
        
        # 1) PR curves
        plots_path = os.path.join(os.getcwd(), 'plots')
        mkdir(plots_path)
        pr_curve = PlotPrecisionRecall(legend=True)
        metrics = herdnet_evaluator._stored_metrics
        for c in range(1, metrics.num_classes):
            rec, pre = metrics.rec_pre_lists(c)
            pr_curve.feed(rec, pre, label=cls_dict[c])
        
        pr_curve.save(os.path.join(plots_path, 'precision_recall_curve.png'))
        
        # 2) metrics per class
        res = herdnet_evaluator.results
        cols = res.columns.tolist()
        str_cls_dict = {str(k): v for k,v in cls_dict.items()}
        str_cls_dict.update({'binary': 'binary'})
        res['species'] = res['class'].map(str_cls_dict)
        res = res[['class', 'species'] + cols[1:]]
        print(res)

        res.to_csv(os.path.join(os.getcwd(), 'metrics_results.csv'), index=False)

        # 3) confusion matrix
        cm = pandas.DataFrame(metrics.confusion_matrix, columns=cls_names, index=cls_names)
        cm.to_csv(os.path.join(os.getcwd(), 'confusion_matrix.csv'))
        print(cm)

        # 4) detections
        detections =  herdnet_evaluator.detections
        detections['species'] = detections['labels'].map(cls_dict)
        detections.to_csv(os.path.join(os.getcwd(), 'detections.csv'), index=False)

    # print(out)


if __name__ == "__main__":

    args = parse(Flags)

    print('\nargs:\n',json.dumps(args.__dict__,indent=2),'\n',flush=True)

    herdnet_val(args)