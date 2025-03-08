from zenml import step
from ..train import HerdnetTrainer, TRANSFORMS
import lightning as L
from ..dataset.handlers import HerdnetPredictDataset
from torch.utils.data import DataLoader


@step
def predict(args,images_path:list[str], imgsz:int, down_ratio:int,
            checkpoint_path:str, work_dir:str, batchsize:int=1, split:str="val",
             accelerator:str="auto", classification_threshold:float=0.25):

    assert split in ["val", "test"]
  
    # Get number of classes
    with open(args.data_config_yaml, 'r') as file:
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
    data = HerdnetPredictDataset(images_path=images_path, 
                                        albu_transforms = TRANSFORMS['val'][0], 
                                        end_transforms = TRANSFORMS['val'][1]
                                )
    data = DataLoader(data, batch_size=batchsize,
                    num_workers=0, shuffle=False, persistent_workers=True)
                                        
    trainer = L.Trainer(accelerator=accelerator,profiler='simple')
    
    out = trainer.predict(model=herdnet_trainer,data)
    
    return out