from zenml import step
from ..train import HerdnetTrainer, HerdnetData
import lightning as L

@step
def predict(imgsz, down_ratio, data_config_yaml, checkpoint_path, work_dir,batchsize=1, split="val", accelerator="auto", classification_threshold=0.25):

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

    trainer = L.Trainer(accelerator=accelerator,profiler='simple')
    
    out = trainer.predict(model=herdnet_trainer,datamodule=datamodule)
    
    return out