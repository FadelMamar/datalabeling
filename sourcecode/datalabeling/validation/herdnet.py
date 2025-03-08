import yaml
import lightning as L

def herdnet_validate(imgsz:int,batchsize:int, down_ratio:int,
                    data_config_yaml:str, checkpoint_path:str, logger,
                    work_dir:str:None, split:str="val", accelerator:str="auto",
                    classification_threshold:float=0.25):

    assert split in ["val", "test"]
  
    # Get number of classes
    with open(data_config_yaml, 'r') as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    num_classes = data_config['nc']+1
    
    
    herdnet_trainer = HerdnetTrainer.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                            args=args,
                                                            herdnet_model_path = None,
                                                            loaded_weights_num_classes=num_classes,
                                                            classification_threshold=classification_threshold,
                                                            ce_weight=None,
                                                            map_location='cpu',
                                                            strict=True,
                                                            work_dir=work_dir)
    # Data
    datamodule = HerdnetData(data_config_yaml=data_config_yaml,
                                patch_size=imgsz,
                                batch_size=batchsize,
                                down_ratio=down_ratio,
                                train_empty_ratio=0.,
                                )

    trainer = L.Trainer(accelerator=accelerator,profiler='simple',logger=logger)
    
    if split == "val":
        out = trainer.validate(model=herdnet_trainer,datamodule=datamodule)
    else:    
        out = trainer.test(model=herdnet_trainer,datamodule=datamodule)
    
    return out