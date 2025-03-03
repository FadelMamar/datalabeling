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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor

def run_ligthning():
    
    args = Arguments()
    args.lr0 = 3e-4
    args.epochs = 50
    args.imgsz = 800
    args.batchsize = 32
    down_ratio = 2
    precision = "16"

    args.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"
    args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    # args.data_config_yaml = r"D:\datalabeling\data\data_config.yaml"
    # args.path_weights = r"D:\datalabeling\models\20220329_HerdNet_Ennedi_dataset_2023.pth"
    
    # loggers and callbacks
    mlf_logger = MLFlowLogger(experiment_name="Herdnet", 
                              run_name="herdnet",
                              tracking_uri=args.mlflow_tracking_uri,
                              log_model=True
                              )
    checkpoint_callback  = ModelCheckpoint(dirpath="./lightning-ckpts",
                                           monitor='val_f1-score',
                                           mode="max",
                                           save_weights_only=True,
                                           save_top_k=1
                                           )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_callback, DeviceStatsMonitor()]
    
    # Data
    datamodule = HerdnetData(data_config_yaml=args.data_config_yaml,
                   patch_size=args.imgsz,
                   batch_size=args.batchsize,
                   down_ratio=down_ratio
                   )
    
    # Training logic
    herndet_trainer = HerdnetTrainer(herdnet_model_path=args.path_weights,
                                    args=args,
                                    ce_weight=None,
                                    work_dir='../.tmp'
                                    )

    # Trainer
    profiler = AdvancedProfiler()
    trainer = L.Trainer(num_sanity_val_steps=10,
                    logger=mlf_logger,
                    max_epochs=args.epochs,
                    precision=precision,
                    callbacks=callbacks,
                    accelerator="gpu",
                    profiler=profiler
                )
    trainer.fit(model=herndet_trainer,
                datamodule=datamodule
              )


def run():
    
    args = Arguments()
    args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    args.lr0 = 3e-4
    args.imgsz = 800
    args.batchsize = 16
    args.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"
    down_ratio = 2
    
    # Data
    datamodule = HerdnetData(data_config_yaml=args.data_config_yaml,
                            patch_size=args.imgsz,
                            batch_size=args.batchsize,
                            down_ratio=down_ratio
                   )
    
    datamodule.setup("fit")
    
    num_classes = datamodule.num_classes

    ce_weights = None #datamodule.get_labels_weights

    losses = [
        {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
        {'loss': CrossEntropyLoss(reduction='mean', weight=ce_weights), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
        ]

    herdnet = HerdNet(pretrained=False,down_ratio=down_ratio,num_classes=4)
    herdnet = LossWrapper(herdnet, losses=losses)
    checkpoint = torch.load(args.path_weights, map_location="cpu",weights_only=True)
    herdnet.load_state_dict(checkpoint['model_state_dict'],strict=True)
    herdnet.model.reshape_classes(num_classes)

    herdnet = herdnet.to('cuda')

    work_dir = '../.tmp'

    lr = 1e-4
    weight_decay = 1e-4
    epochs = 30
    patch_size = 800

    optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)

    metrics = PointsMetrics(radius=20, num_classes=num_classes)

    stitcher = HerdNetStitcher(
        model=herdnet, 
        size=(patch_size,patch_size), 
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
    
    herdnet = trainer.start(warmup_iters=30, checkpoints='best', select='max', validate_on='f1_score')


if __name__ == "__main__":
    run()
    # run_ligthning()
    pass