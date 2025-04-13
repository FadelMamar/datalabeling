from datargs import parse
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Sequence
import json
from datalabeling.train.herdnet import HerdnetData, HerdnetTrainer
import lightning as L
import pandas as pd
import numpy as np
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
    splits: Sequence[str] = ("val",)

    # weights
    weights: str = None

    # validation enginge
    engine: str = "original"  # original, lightning

    # inference
    batch_size: int = 16
    imgsz: int = 800
    device: str = "cuda"
    down_ratio: int = 2
    eval_radius: int = 20

    # logging
    save_csv: bool = False
    name: str = "herdnet"
    save_dir: str = "runs_herdnet"
    project_name: str = "Herdnet"
    plots: bool = False
    log_mlflow: bool = False
    print_freq: int = 100


def plot_conf_matrix(array: np.ndarray, labels: list, save_dir: str):
    """Plots confusion matrix

    Args:
        array (np.ndarray): confusion matrix computed by scikit-learn
        labels (list): labels without background class
        save_dir (str): directory to save plots
    """
    import seaborn
    import matplotlib.pyplot as plt
    import json

    array = array.T  # get matrix like (Predicted,True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    seaborn.set_theme(font_scale=1.0)  # for label size
    nc = len(labels)
    ticklabels = labels
    seaborn.heatmap(
        array,
        ax=ax,
        annot=nc < 30,
        annot_kws={"size": 8},
        cmap="Blues",
        fmt=".0f",
        square=True,
        vmin=0.0,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    ).set_facecolor((1, 1, 1))
    title = "Confusion Matrix"
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
    fig.savefig(plot_fname, dpi=250)

    # Compute precision, recall, f1-score
    results_per_cls = dict()
    for i, label in enumerate(labels):
        if label == "background":
            break
        tp = array[i, i]
        actual_positive = array[:, i].sum()
        predicted_positive = array[i, :].sum()
        # fp = predicted_positive - tp
        # fn = actual_positive - tp
        precision = tp / (predicted_positive + 1e-8)
        recall = tp / (actual_positive + 1e-8)
        f1score = 2 * precision * recall / (precision + recall + 1e-8)

        results = dict(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1score=round(f1score, 4),
        )
        results_per_cls[label] = results
        print(f"results for {label} : ", results, end="\n")

    with open(Path(save_dir) / "metrics_per_class.json", "w") as file:
        json.dump(results_per_cls, file, indent=2)


def herdnet_val(args: Flags):
    assert args.engine in ["original", "lightning"]

    args.save_dir = Path(args.save_dir) / args.name

    args.save_dir.mkdir(exist_ok=True, parents=True)

    # Get number of classes
    with open(args.data_config, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    num_classes = data_config["nc"] + 1
    cls_dict = {k + 1: v for k, v in data_config["names"].items()}
    cls_names = [cls_dict[i + 1] for i in range(len(cls_dict))]

    # set model
    if args.log_mlflow:
        mlf_logger = MLFlowLogger(
            experiment_name="Herdnet",
            run_name="herdnet-validate",
            tracking_uri=args.mlflow_tracking_uri,
            log_model=True,
        )
    else:
        mlf_logger = None

    map_location = torch.device("cpu")
    if torch.cuda.is_available():
        map_location = torch.device("cuda")

    # Data
    datamodule = HerdnetData(
        data_config_yaml=args.data_config,
        patch_size=args.imgsz,
        batch_size=args.batch_size,
        down_ratio=args.down_ratio,
        train_empty_ratio=0.0,
    )

    if args.engine != "original":
        herdnet_trainer = HerdnetTrainer.load_from_checkpoint(
            checkpoint_path=args.weights,
            data_config_yaml=args.data_config,
            losses=[],
            lr=None,
            weight_decay=None,
            herdnet_model_path=None,
            loaded_weights_num_classes=num_classes,
            ce_weight=None,
            eval_radius=args.eval_radius,
            map_location=map_location,
            strict=True,
            work_dir=args.save_dir,
        )
    else:
        herdnet_trainer = HerdnetTrainer(
            herdnet_model_path=args.weights,
            data_config_yaml=args.data_config,
            lr=0.0,
            weight_decay=0.0,
            down_ratio=args.down_ratio,
            loaded_weights_num_classes=datamodule.num_classes,
            ce_weight=None,
            work_dir=args.save_dir,
            losses=None,
            eval_radius=args.eval_radius,
            load_state_dict_strict=True,
        )

    # Predict
    # images_path = os.path.join(data_config['path'],data_config['test'][0])
    # images_path = list(Path(images_path).glob('*'))
    # datamodule.set_predict_dataset(images_path=images_path,batchsize=1)

    trainer = L.Trainer(accelerator="auto", logger=mlf_logger)

    for split in args.splits:
        if split == "val":
            datamodule.setup("validate")
            herdnet_evaluator = HerdNetEvaluator(
                model=herdnet_trainer.model,
                dataloader=datamodule.val_dataloader(),
                print_freq=args.print_freq,
                metrics=herdnet_trainer.metrics_val,
                stitcher=herdnet_trainer.stitcher,
                work_dir=args.save_dir,
                header="validation",
                lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 3.0, "neg_ts": 0.1},
            )
            herdnet_trainer.herdnet_evaluator = herdnet_evaluator
            if args.engine != "original":
                trainer.validate(model=herdnet_trainer, datamodule=datamodule)

        elif split == "test":
            datamodule.setup("test")
            herdnet_evaluator = HerdNetEvaluator(
                model=herdnet_trainer.model,
                dataloader=datamodule.test_dataloader(),
                metrics=herdnet_trainer.metrics_val,
                print_freq=args.print_freq,
                stitcher=herdnet_trainer.stitcher,
                work_dir=args.save_dir,
                header="test",
                lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 3.0, "neg_ts": 0.1},
            )
            herdnet_trainer.herdnet_evaluator = herdnet_evaluator
            if args.engine != "original":
                trainer.test(model=herdnet_trainer, datamodule=datamodule)

        else:
            raise NotImplementedError

        # Run evaludation
        if args.engine == "original":
            herdnet_evaluator.evaluate(wandb_flag=False, viz=False)

        # Save results
        print("Saving the results ...")

        # 1) PR curves
        plots_path = os.path.join(args.save_dir, "plots")
        mkdir(plots_path)
        pr_curve = PlotPrecisionRecall(legend=True)
        metrics = herdnet_evaluator._stored_metrics
        for c in range(1, metrics.num_classes):
            rec, pre = metrics.rec_pre_lists(c)
            pr_curve.feed(rec, pre, label=cls_dict[c])

        pr_curve.save(os.path.join(plots_path, "precision_recall_curve.png"))

        # 2) metrics per class
        res = herdnet_evaluator.results
        cols = res.columns.tolist()
        str_cls_dict = {str(k): v for k, v in cls_dict.items()}
        str_cls_dict.update({"binary": "binary"})
        res["species"] = res["class"].map(str_cls_dict)
        res = res[["class", "species"] + cols[1:]]
        print(res)

        res.to_csv(os.path.join(args.save_dir, "metrics_results.csv"), index=False)

        # 3) confusion matrix
        cm = pd.DataFrame(metrics.confusion_matrix, columns=cls_names, index=cls_names)
        cm.to_csv(os.path.join(args.save_dir, "confusion_matrix.csv"))
        print(cm)
        # print(metrics._confusion_matrix)

        plot_conf_matrix(
            metrics.confusion_matrix, labels=cls_names, save_dir=plots_path
        )

        # 4) detections
        # detections = list()
        # for dataset in herdnet_evaluator.dataloader.dataset.datasets:
        #     img_names = dataset._img_names
        #     dets = herdnet_evaluator._stored_metrics.detections
        #     for det in dets:
        #         det['images'] = img_names[det['images']]
        #     detections.append(pd.DataFrame(data = dets))

        # detections = pd.concat(detections)
        # print(detections.head())
        # detections['species'] = detections['label'].map(cls_dict)
        # detections.to_csv(os.path.join(args.save_dir, 'detections.csv'), index=False)


if __name__ == "__main__":
    args = parse(Flags)

    print("\nargs:\n", json.dumps(args.__dict__, indent=2), "\n", flush=True)

    herdnet_val(args)
