



def ultralytics_val():
    from ultralytics import YOLO
    # from pathlib import Path
    from datalabeling.train import remove_label_cache
    
    # Getting results for yolov12s : Detection and Identification
    paths = ["../runs/mlflow/140168774036374062/e0ea49b51ce34cfe9de6b482a2180037/artifacts/weights/best.pt", # Identification model weights
            "../runs/mlflow/140168774036374062/a59eda79d9444ff4befc561ac21da6b4/artifacts/weights/best.pt" # Detection model weights
            ]

    dataconfigs = [
                    r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml",
                    r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification-detection.yaml"
                ]

    imgsz = 800
    iou_threshold=0.45
    conf_threshold=0.235
    splits = [
            "val", 
            "test",
            ]

    # remove label.cache files
    for dataconfig in dataconfigs:
        remove_label_cache(data_config_yaml=dataconfig)

    for split in splits:
        for path,dataconfig in zip(paths,dataconfigs):
            print("\n",'-'*20,split,'-'*20)
            model = YOLO(path)
            model.info()
            
            # Customize validation settings
            validation_results = model.val(data=dataconfig,
                                            imgsz=imgsz,
                                            batch=64,
                                            split=split,
                                            conf=conf_threshold,
                                            iou=iou_threshold,
                                            device="cuda"
                                        )


def herdnet_val():

    from datalabeling.train.herdnet import HerdnetData, HerdnetTrainer
    from datalabeling.arguments import Arguments
    import lightning as L
    import os, yaml
    from pathlib import Path
    import torch
    import random
    import pandas as pd
    from lightning.pytorch.loggers import MLFlowLogger


    # lowering matrix multiplication precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    args = Arguments()
    args.imgsz = 800
    args.batchsize = 32
    down_ratio = 2
    
    # =============================================================================
    #     Identification
    # =============================================================================
    
    # args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    # checkpoint_path = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools\lightning-ckpts\epoch=11-step=1740.ckpt"
    # num_classes = 7
    
    # =============================================================================
    #     # Pretrained
    # =============================================================================
    # args.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"
    
    
    # =============================================================================
    #     Detection
    # =============================================================================
    args.data_config_yaml = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification-detection.yaml"
    checkpoint_path = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\mlartifacts\934358897506090439\1f0f0be9be1a406c8df8978331a99915\artifacts\epoch=2-step=1815\epoch=2-step=1815.ckpt"
    num_classes = 2

    # Example: load Prediction images
    # with open(args.data_config_yaml, 'r') as file:
    #     data_config = yaml.load(file, Loader=yaml.FullLoader)
    
    
    # set model
    mlf_logger = MLFlowLogger(experiment_name="Herdnet",
                            run_name="herdnet-validate",
                            tracking_uri=args.mlflow_tracking_uri,
                            log_model=True
                            )
    herdnet_trainer = HerdnetTrainer.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                            args=args,
                                                            herdnet_model_path = None,
                                                            loaded_weights_num_classes=num_classes,
                                                            classification_threshold=0.25,
                                                            ce_weight=None,
                                                            map_location='cpu',
                                                            strict=True,
                                                            work_dir='../.tmp')

    # Data
    datamodule = HerdnetData(data_config_yaml=args.data_config_yaml,
                                patch_size=args.imgsz,
                                batch_size=args.batchsize,
                                down_ratio=down_ratio,
                                train_empty_ratio=0.,
                                )
    # Validation
    # datamodule.setup('fit')
    
    # Predict
    # images_path = os.path.join(data_config['path'],data_config['test'][0])
    # images_path = list(Path(images_path).glob('*'))
    # datamodule.set_predict_dataset(images_path=images_path,batchsize=1)
    
    trainer = L.Trainer(accelerator="auto",profiler='simple',logger=mlf_logger)
    
    # out = trainer.validate(model=herdnet_trainer,datamodule=datamodule)
    
    out = trainer.test(model=herdnet_trainer,datamodule=datamodule)
    
    # out = trainer.predict(model=herdnet_trainer, datamodule=datamodule,)

if __name__ == "__main__":

    # ultralytics_val()
    
    herdnet_val()
    

