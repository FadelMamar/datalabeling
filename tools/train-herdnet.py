from animaloc.models import HerdNet
import torch
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetEvaluator
from datalabeling.train.herdnet import HerdnetData, HerdnetTrainer
from datalabeling.arguments import Arguments
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pathlib import Path


def run_ligthning(args: Arguments):
    import logging
    import segmentation_models_pytorch as smp

    logger = logging.getLogger("mlflow")
    logger.setLevel(logging.DEBUG)

    # lowering matrix multiplication precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # args = Arguments()
    # args.lr0 = 1e-4
    args.lrf = 0.1
    # args.epochs = 30
    args.imgsz = 800
    args.batchsize = 16
    down_ratio = 2
    precision = "bf16-mixed"
    empty_ratio = 0.0
    args.patience = 10
    args.cl_batch_size = 16
    args.cl_lr0s = [3e-4, 1e-4, 5e-5]
    args.cl_ratios = [0, 1, 2.5]
    args.cl_freeze = [0.0, 0.5, 0.75]
    args.cl_epochs = [30, 10, 7]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.project_name = "Herdnet"

    accelerator = "auto"
    detect_anomaly = False

    normalization = "standard"  # "standard min_max
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean=(0., 0., 0.)
    # std=(1., 1., 1.)

    check_val_every_n_epoch = 3
    num_sanity_val_steps = 10

    work_dir = Path("runs_herdnet")  # for HerdNet Trainer
    work_dir.mkdir(exist_ok=True, parents=False)

    args.path_weights = r"base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"  # initialization
    loaded_weights_num_classes = 4  # for ennedi weights
    # args.data_config_yaml = r"configs\yolo_configs\dataset_identification.yaml"
    # args.run_name = "herdnet-Identif"

    # args.data_config_yaml = r"D:\datalabeling\data\data_config.yaml"
    # args.path_weights = r"D:\datalabeling\models\20220329_HerdNet_Ennedi_dataset_2023.pth"

    checkpoint_path = None
    # checkpoint_path = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\mlartifacts\934358897506090439\336b6791130f4873903e924d26beccad\artifacts\epoch=4-step=450\epoch=4-step=450.ckpt"
    # loaded_weights_num_classes = 2  # num classes includes background

    # get cross entropy loss weights
    # Data
    # datamodule = HerdnetData(
    #     data_config_yaml=args.data_config_yaml,
    #     patch_size=args.imgsz,
    #     batch_size=args.batchsize,
    #     down_ratio=down_ratio,
    #     train_empty_ratio=empty_ratio,
    # )
    # datamodule.setup("fit")
    # ce_weight = datamodule.get_labels_weights.clamp(min=1.1).log()
    # ce_weight[0] = 0.01
    # ce_weight = ce_weight.to(device)

    ce_weight = None
    logger.info(f"cross entropy loss class importance weights: {ce_weight}")
    datamodule = None

    if checkpoint_path is not None:
        herdnet_trainer = HerdnetTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            lr=args.lr0,
            weight_decay=args.weight_decay,
            data_config_yaml=args.data_config_yaml,
            herdnet_model_path=None,  # should be None!
            loaded_weights_num_classes=loaded_weights_num_classes,
            ce_weight=ce_weight,
            map_location="cpu",
            strict=True,
            work_dir=work_dir,
        )

        print(f"\nLoading checkpoint at {checkpoint_path}\n")
    else:
        losses = None  # uses the default
        # Training logic
        herdnet_trainer = HerdnetTrainer(
            herdnet_model_path=args.path_weights,
            data_config_yaml=args.data_config_yaml,
            lr=args.lr0,
            weight_decay=args.weight_decay,
            loaded_weights_num_classes=loaded_weights_num_classes,
            ce_weight=ce_weight,
            work_dir=work_dir,
            losses=losses,
            load_state_dict_strict=True,
        )

    # continuous learning
    for empty_ratio, lr, freeze_ratio, epochs in zip(
        args.cl_ratios, args.cl_lr0s, args.cl_freeze, args.cl_epochs
    ):
        args.run_name = (
            args.run_name + f"-emptyRatio_{empty_ratio}-freezeRatio_{freeze_ratio}"
        )
        args.cl_save_dir = work_dir / args.run_name
        args.cl_save_dir.mkdir(parents=True, exist_ok=True)

        # loggers and callbacks
        mlf_logger = MLFlowLogger(
            experiment_name=args.project_name,
            run_name=args.run_name,
            tracking_uri=args.mlflow_tracking_uri,
            log_model=True,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.cl_save_dir,
            monitor="val_f1-score",
            mode="max",
            save_weights_only=True,
            save_last=True,
            save_top_k=1,
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            checkpoint_callback,
            lr_callback,
            EarlyStopping(
                monitor="val_f1-score",
                patience=args.patience,
                min_delta=1e-4,
                mode="max",
            ),
        ]

        herdnet_trainer.hparams.lr = lr
        herdnet_trainer.hparams.epochs = epochs
        herdnet_trainer.hparams.lrr = args.lrf

        # Freeze params
        num_layers = len(list(herdnet_trainer.parameters()))
        for idx, param in enumerate(herdnet_trainer.parameters()):
            if idx / num_layers < freeze_ratio:
                param.requires_grad = False
            else:
                break
        logger.info(f"\n{int(num_layers * freeze_ratio)} layers have been frozen.\n")

        # Data
        datamodule = HerdnetData(
            data_config_yaml=args.data_config_yaml,
            patch_size=args.imgsz,
            batch_size=args.cl_batch_size,
            down_ratio=down_ratio,
            train_empty_ratio=empty_ratio,
            normalization=normalization,  #
            mean=mean,
            std=std,
        )

        # Trainer
        trainer = L.Trainer(
            num_sanity_val_steps=num_sanity_val_steps,
            logger=mlf_logger,
            max_epochs=epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            # accumulate_grad_batches=max(int(64 / args.batchsize), 1),
            precision=precision,
            callbacks=callbacks,
            gradient_clip_val=10,
            gradient_clip_algorithm="value",
            detect_anomaly=detect_anomaly,
            accelerator=accelerator,
        )
        trainer.fit(
            model=herdnet_trainer,
            datamodule=datamodule,
        )
        # trainer.validate(model=herdnet_trainer,
        #             datamodule=datamodule,
        #             )
        # Reset param.requires_grad
        for param in herdnet_trainer.parameters():
            param.requires_grad = True


def run(args: Arguments):
    down_ratio = 2
    empty_ratio = args.cl_ratios[0]
    valid_freq = 4
    work_dir = r"runs_herdnet"  # for HerdNet Trainer
    work_dir = Path(work_dir) / (args.run_name)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Data
    datamodule = HerdnetData(
        data_config_yaml=args.data_config_yaml,
        patch_size=args.imgsz,
        batch_size=args.batchsize,
        down_ratio=down_ratio,
        train_empty_ratio=empty_ratio,
    )

    datamodule.setup("fit")

    # check val dataloaders
    # for img_val,targets_val in tqdm(datamodule.val_dataloader(), desc="Val dataloader check"):
    #     pass
    # for img_tr,targets_tr in tqdm(datamodule.train_dataloader(),desc="Train dataloader check"):
    #     pass

    num_classes = datamodule.num_classes

    # ce_weights = datamodule.get_labels_weights.to(args.device)
    ce_weights = None

    losses = [
        {
            "loss": FocalLoss(reduction="mean"),
            "idx": 0,
            "idy": 0,
            "lambda": 1.0,
            "name": "focal_loss",
        },
        {
            "loss": CrossEntropyLoss(reduction="mean", weight=ce_weights),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]

    # Load model
    herdnet = HerdNet(
        pretrained=False, down_ratio=down_ratio, num_classes=args.herdnet_num_classes
    )
    herdnet = LossWrapper(herdnet, losses=losses)
    checkpoint = torch.load(
        args.path_weights, map_location=args.device, weights_only=True
    )
    success = herdnet.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"Loading weights from {args.path_weights} with success: {success}")

    herdnet.model.reshape_classes(num_classes)
    herdnet = herdnet.to(args.device)

    optimizer = Adam(
        params=herdnet.parameters(), lr=args.lr0, weight_decay=args.weight_decay
    )

    metrics = PointsMetrics(radius=20, num_classes=num_classes)

    # stitcher = HerdNetStitcher(
    #     model=herdnet,
    #     size=(patch_size, patch_size),
    #     batch_size=1,
    #     overlap=160,
    #     down_ratio=down_ratio,
    #     reduction='mean'
    # )
    stitcher = None

    evaluator = HerdNetEvaluator(
        model=herdnet,
        dataloader=datamodule.val_dataloader(),
        metrics=metrics,
        device_name=args.device,
        print_freq=100,
        stitcher=stitcher,
        work_dir=work_dir,
        header="validation",
    )

    # check
    # evaluator.evaluate()

    trainer = Trainer(
        model=herdnet,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=None,
        valid_freq=valid_freq,
        print_freq=100,
        lr_milestones=[
            20,
        ],
        optimizer=optimizer,
        auto_lr=True,
        device_name=args.device,
        num_epochs=args.epochs,
        evaluator=evaluator,
        work_dir=work_dir,
    )

    herdnet = trainer.start(
        warmup_iters=50, checkpoints="best", select="max", validate_on="f1_score"
    )


if __name__ == "__main__":
    from datargs import parse
    import mlflow

    args = parse(Arguments)

    # training using lightning
    # run_ligthning(args)

    # training using original pipeline
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.project_name)
    mlflow.pytorch.autolog()

    args.run_name += f"_emptyRatio_{args.cl_ratios[0]}"

    with mlflow.start_run(
        run_name=args.run_name,
        experiment_id=mlflow.get_experiment_by_name(args.project_name).experiment_id,
    ):
        mlflow.log_params(vars(args))

        run(args)
