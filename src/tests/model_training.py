from datalabeling.common.config import TrainingConfig
from datalabeling.common.pipeline import ModelTraining, Pipeline


if __name__ == "__main__":
    ## Training configs
    training_cfg = TrainingConfig()
    training_cfg.herdnet_work_dir = r"D:\datalabeling\.tmp"
    training_cfg.run_name = "debug"
    training_cfg.imgsz = 640
    training_cfg.batchsize = 4
    training_cfg.epochs = 15
    training_cfg.herndet_empty_ratio = 0.0
    training_cfg.herdnet_lr_milestones = [15, 25]
    training_cfg.herdnet_val_batchsize = 4

    training_cfg.yolo_yaml = (
        r"D:\datalabeling\configs\yolo_configs\data\data_config.yaml"
    )
    training_cfg.yolo_arch_yaml = (
        r"D:\datalabeling\configs\yolo_configs\models\yolov8-ghost-p2.yaml"
    )
    training_cfg.path_weights = None

    training_cfg.ultralytics_pos_weight = 10.0

    training_step = ModelTraining(
        training_cfg=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        model_type="ultralytics",
    )

    pipe = Pipeline(
        steps=[
            training_step,
        ]
    )

    ## uncomment to run
    # pipe.run()
