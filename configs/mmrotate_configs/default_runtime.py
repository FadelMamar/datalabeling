# yapf:disable
# loggers: https://mmcv.readthedocs.io/en/master/_modules/mmcv/runner/hooks/logger/wandb.html
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'), # Uncomment this line to enable Tensorboard
        dict(type='MlflowLoggerHook'), # Uncomment this line to enable Mlflow
        # dict(type='WandbLoggerHook'), # Uncomment this line to enable wandb
    ])
# yapf:enable

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
# https://mmrotate.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow
workflow = [("train", 1), ("val", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"
