from animaloc.models import HerdNet
import torch
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator


from datalabeling.train.herdnet import HerdnetData, HerdnetTrainer
from datalabeling.arguments import Arguments
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


def run_ligthning():

    import logging

    logger = logging.getLogger("mlflow")
    logger.setLevel(logging.DEBUG)

    # lowering matrix multiplication precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    args = Arguments()
    args.lr0 = 3e-4
    args.epochs = 15
    args.imgsz = 800
    args.batchsize = 32
    down_ratio = 2
    precision = "16-mixed"
    # empty_ratio = 0.
    args.patience = 10
    cl_lr = [3e-4, 5e-5, 5e-5]
    empty_ratios = [1, 2.5, 7.5]
    freeze_layers = [0., 0.5, 0.75]

    args.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"  # initialization
    args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    # args.data_config_yaml = r"D:\datalabeling\data\data_config.yaml"
    # args.path_weights = r"D:\datalabeling\models\20220329_HerdNet_Ennedi_dataset_2023.pth"

    checkpoint_path = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools\lightning-ckpts\epoch=23-step=2040.ckpt"

    # loggers and callbacks
    mlf_logger = MLFlowLogger(experiment_name="Herdnet",
                              run_name="herdnet",
                              tracking_uri=args.mlflow_tracking_uri,
                              log_model=True
                              )
    checkpoint_callback = ModelCheckpoint(dirpath="./lightning-ckpts",
                                          monitor='val_f1-score',
                                          mode="max",
                                          save_weights_only=True,
                                          save_top_k=1
                                          )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback,
                 lr_callback,
                 EarlyStopping(monitor='val_f1-score',
                               patience=args.patience,
                               min_delta=1e-4,
                               mode="max")
                 ]

    # Training logic
    herdnet_trainer = HerdnetTrainer(herdnet_model_path=args.path_weights,
                                     args=args,
                                     ce_weight=None,
                                     work_dir='../.tmp',
                                     # set to False if pretrained weights are being loaded properly due to classification head
                                     load_state_dict_strict=True,
                                     )
    if checkpoint_path is not None:
        herdnet_trainer = HerdnetTrainer.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                              args=args,
                                                              ce_weight=None,
                                                              work_dir='../.tmp')

        print(f"\nLoading checkpoint at {checkpoint_path}\n")

    for empty_ratio, lr, freeze_ratio in zip(empty_ratios, cl_lr, freeze_layers):

        herdnet_trainer.args.lr0 = lr

        # freeze params
        num_layers = len(list(herdnet_trainer.parameters()))
        for idx, param in enumerate(herdnet_trainer.parameters()):
            if idx/num_layers < freeze_ratio:
                param.requires_grad = False
            else:
                break

        # Data
        datamodule = HerdnetData(data_config_yaml=args.data_config_yaml,
                                 patch_size=args.imgsz,
                                 batch_size=args.batchsize,
                                 down_ratio=down_ratio,
                                 train_empty_ratio=empty_ratio
                                 )

        # Trainer
        trainer = L.Trainer(num_sanity_val_steps=10,
                            logger=mlf_logger,
                            max_epochs=args.epochs,
                            accumulate_grad_batches=max(
                                int(64/args.batchsize), 1),
                            precision=precision,
                            callbacks=callbacks,
                            accelerator="gpu",
                            )
        trainer.fit(model=herdnet_trainer,
                    datamodule=datamodule,
                    )


def run():

    args = Arguments()
    args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    args.lr0 = 3e-4
    args.imgsz = 800
    args.batchsize = 16
    args.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"
    down_ratio = 2
    empty_ratio = 0.

    # Data
    datamodule = HerdnetData(data_config_yaml=args.data_config_yaml,
                             patch_size=args.imgsz,
                             batch_size=args.batchsize,
                             down_ratio=down_ratio,
                             train_empty_ratio=empty_ratio
                             )

    datamodule.setup("fit")

    num_classes = datamodule.num_classes

    ce_weights = None  # datamodule.get_labels_weights

    losses = [
        {'loss': FocalLoss(reduction='mean'), 'idx': 0,
         'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
        {'loss': CrossEntropyLoss(reduction='mean', weight=ce_weights),
         'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

    herdnet = HerdNet(pretrained=False, down_ratio=down_ratio, num_classes=4)
    herdnet = LossWrapper(herdnet, losses=losses)
    checkpoint = torch.load(
        args.path_weights, map_location="cpu", weights_only=True)
    herdnet.load_state_dict(checkpoint['model_state_dict'], strict=True)
    herdnet.model.reshape_classes(num_classes)

    herdnet = herdnet.to('cuda')

    work_dir = '../.tmp'

    lr = 1e-4
    weight_decay = 1e-4
    epochs = 30
    patch_size = 800

    optimizer = Adam(params=herdnet.parameters(),
                     lr=lr, weight_decay=weight_decay)

    metrics = PointsMetrics(radius=20, num_classes=num_classes)

    stitcher = HerdNetStitcher(
        model=herdnet,
        size=(patch_size, patch_size),
        batch_size=1,
        overlap=160,
        down_ratio=down_ratio,
        reduction='mean'
    )

    evaluator = HerdNetEvaluator(
        model=herdnet,
        dataloader=datamodule.val_dataloader(),
        metrics=metrics,
        stitcher=stitcher,
        work_dir=work_dir,
        header='validation'
    )

    trainer = Trainer(
        model=herdnet,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=None,
        lr_milestones=[20,],
        optimizer=optimizer,
        auto_lr=True,
        num_epochs=epochs,
        evaluator=evaluator,
        work_dir=work_dir
    )

    herdnet = trainer.start(
        warmup_iters=30, checkpoints='best', select='max', validate_on='f1_score')


if __name__ == "__main__":

    # run()

    run_ligthning()

    pass
