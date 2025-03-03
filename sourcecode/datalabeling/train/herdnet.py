import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
import albumentations as A
import yaml
from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT
import pandas as pd
import albumentations as A
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from ..arguments import Arguments
from animaloc.models import HerdNet, load_model
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
import numpy as np
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir
from PIL import Image


def check_label_format(loaded_df:pd.DataFrame)->str:
    """checks label format

    Args:
        loaded_df (pd.DataFrame): target values

    Raises:
        NotImplementedError: when the format is not yolo or yolo-obb

    Returns:
        str: yolo or yolo-obb
    """

    num_features = len(loaded_df.columns)

    if num_features == 5:
        return "yolo"
    elif num_features == 9:
        return "yolo-obb"
    else:
        raise NotImplementedError(f"The number of features ({num_features}) in the label file is wrong. Check yolo or yolo-obb format from ultralytics.")


def get_groundtruth(yolo_labels_dir, save_path:str=None,load_gt_csv:str=None):

    if load_gt_csv is not None:
        return pd.read_csv(load_gt_csv)

    cols1 = ['id','x1','y1','x2','y2','x3','y3','x4','y4']
    cols2 = ['id','x','y','w','h']

    if not Path(yolo_labels_dir).exists():
        raise FileNotFoundError('Directory does not exist.')

    # Iterate through labels
    dfs = list()
    for label_path in tqdm(Path(yolo_labels_dir).glob("*.txt"),desc="Getting groundtruths"):
        df = pd.read_csv(label_path,sep=' ',header=None)

        image_path = Path(str(label_path).replace('labels','images')).with_suffix(".jpg")
        img = Image.open(image_path)
        img_width, img_height = img.size

        label_format = check_label_format(df)

        if label_format == 'yolo':
            df.columns = cols2
            df.loc[:,'x'] = df['x']*img_width
            df.loc[:,'y'] = df['y']*img_height
            df.loc[:,'w'] = df['w']*img_width
            df.loc[:,'h'] = df['h']*img_height
        else: # yolo-obb
            df.columns = cols1
            df['x'] = (df['x1'] + df['x2'])*img_width*0.5
            df['y'] = (df['y1'] + df['y4'])*img_height*0.5
            df['w'] = (df['x2'] - df['x1'])*img_width
            df['h'] = (df['y4'] - df['y1'])*img_height
        df['images'] = str(image_path)
        df.rename(columns={'id':'labels'}, inplace=True)
        dfs.append(df)
        img.close()
    
    dfs = pd.concat(dfs)
    assert dfs["labels"].min() == 0, "Check yolo label format."
    dfs["labels"] = dfs["labels"] + 1 # shift to range [1,num_classes] instead of [0,num_classes[
    if save_path is not None:
        dfs.to_csv(save_path, index=False)

    return dfs


def load_dataset(data_config_yaml:str,
                 split:str,
                 transforms:dict):

    with open(data_config_yaml,'r') as file:
        data_config = yaml.load(file,Loader=yaml.FullLoader)
    datasets = list()
    df_gts = list()
    root = data_config['path']
    for img_dir in tqdm(data_config[split],desc="concatenating datasets"):
        img_dir = os.path.join(root,img_dir)
        # path_to_csv = Path(img_dir).parent/"gt.csv"
        df = get_groundtruth(yolo_labels_dir=img_dir.replace("images","labels"), 
                             save_path=None,
                             load_gt_csv=None # path_to_csv
                            )        
        dataset = CSVDataset(
                            csv_file = df,
                            root_dir = img_dir,
                            albu_transforms = transforms[split][0],
                            end_transforms = transforms[split][1]
                            )
        datasets.append(dataset)
        df_gts.append(df)
    
    return ConcatDataset(datasets=datasets), pd.concat(df_gts)


class HerdnetData(L.LightningDataModule):

    def __init__(self, data_config_yaml:str, patch_size:int, down_ratio:int=2, batch_size: int = 32, transforms:dict[str,tuple]=None):

        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.data_config_yaml = data_config_yaml
        self.transforms = transforms

        # Get number of classes
        with open(data_config_yaml,'r') as file:
            data_config = yaml.load(file,Loader=yaml.FullLoader)
            self.num_classes = data_config['nc'] + 1
        
        if self.transforms is None:
            self.transforms = {}
            self.transforms['train'] = ([    A.VerticalFlip(p=0.5), 
                                        A.HorizontalFlip(p=0.5),
                                        A.RandomRotate90(p=0.5),
                                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                                        A.Blur(blur_limit=15, p=0.2),
                                        A.Normalize(p=1.0)
                                    ], 
                                    [
                                        MultiTransformsWrapper([
                                        FIDT(num_classes=self.num_classes, down_ratio=down_ratio),
                                        PointsToMask(radius=2, 
                                                    num_classes=self.num_classes, 
                                                    squeeze=True, 
                                                    down_ratio=int(patch_size//(16*patch_size/512))
                                                    )
                                    ])]
                                )
            self.transforms['val'] = (
                                        [#A.Resize(patch_size,patch_size,p=1.0),
                                        A.Normalize(p=1.0)
                                        ],
                                        [
                                        DownSample(down_ratio=down_ratio, anno_type='point'),
                                        ]
                                )
            self.transforms['test'] = self.transforms['val']
    
    @property
    def get_labels_weights(self,):
        if self.num_classes == 2:
            return [1.0,1.0]
        weights = 1/(self.df_train_labels_freq + 1e-6)
        weights = [1.0] + weights.to_list()
        assert len(weights) == self.num_classes, "Check for inconsistencies."
        return torch.Tensor(weights)

    def setup(self, stage: str):

        if stage == "fit":
            # train
            self.train_dataset, df_train_labels = load_dataset(data_config_yaml=self.data_config_yaml,
                                                        split='train',
                                                        transforms=self.transforms,
                                                    )
            self.df_train_labels_freq = df_train_labels['labels'].value_counts().sort_index()/len(df_train_labels)
            # val
            self.val_dataset, df_val_labels = load_dataset(data_config_yaml=self.data_config_yaml,
                                                        split='val',
                                                        transforms=self.transforms,
                                                    )
            self.df_val_labels_freq = df_val_labels['labels'].value_counts().sort_index()/len(df_val_labels)
            
        elif stage == "test":
            self.test_dataset, _ = load_dataset(data_config_yaml=self.data_config_yaml,
                                                        split='test',
                                                        transforms=self.transforms,
                        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True,collate_fn=None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1,shuffle=False,collate_fn=None)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False,collate_fn=None)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


class HerdnetTrainer(L.LightningModule):
    def __init__(self,
                 herdnet_model_path:str, 
                 args:Arguments,
                 work_dir:str,
                 ce_weight:list=None):

        super().__init__()
        
        self.args = args
        self.work_dir = work_dir
        
        # Get number of classes
        with open(self.args.data_config_yaml,'r') as file:
            data_config = yaml.load(file,Loader=yaml.FullLoader)
            num_classes = data_config['nc'] + 1 # including a class for background
        
        ce_weight = torch.Tensor(ce_weight) if (ce_weight is not None) else None
        losses = [
                    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
                    {'loss': CrossEntropyLoss(reduction='mean', weight=ce_weight), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
                ]
        
        self.model = HerdNet(pretrained=False,down_ratio=2,num_classes=4)
        self.model = LossWrapper(self.model, losses=losses,mode="both")
        checkpoint = torch.load(herdnet_model_path, map_location="cpu",weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.model.reshape_classes(num_classes)

        # metrics
        radius = 20
        self.metrics_val =  PointsMetrics(radius=radius, num_classes=num_classes)
        self.metrics_test =  PointsMetrics(radius=radius, num_classes=num_classes)

        self.metrics = {"val":self.metrics_val, "test":self.metrics_test}

        self.stitcher = None
        # self.stitcher = HerdNetStitcher(
        #                 model=herdnet, 
        #                 size=(self.args.imgsz,self.args.imgsz), 
        #                 overlap=160, 
        #                 down_ratio=down_ratio, 
        #                 reduction='mean'
        #                 )
        

        # work_dir
        self.herdet_evaluator = HerdNetEvaluator(
            model=self.model, 
            dataloader=DataLoader(dataset=[None,None],batch_size=1), 
            metrics=self.metrics_val, 
            stitcher=self.stitcher, 
            work_dir=self.work_dir, 
            header='validation'
            )
    
        
    def shared_step(self,stage, batch, batch_idx):

        images, targets = batch

        # compute losses
        if stage == "train":
            predictions, loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            self.log(loss_dict)
            
        else:
            predictions, _ = self.model(images)
            loss = torch.zeros([1,]).to(images.device)

            # compute metrics
            output = self.herdet_evaluator.prepare_feeding(targets=targets, output=predictions)
            iter_metrics = self.metrics[stage]
            iter_metrics.feed(**output)
            
        return loss
    
    
    def on_validation_epoch_end(self,):
        stage="val"
        iter_metrics = self.metrics[stage]
        iter_metrics.aggregate() # aggregate results
        self.log(f'{stage}_recall', round(iter_metrics.recall(),3))
        self.log(f'{stage}_precision', round(iter_metrics.precision(),3))
        self.log(f'{stage}_f1-score', round(iter_metrics.fbeta_score(),3))
        self.log(f'{stage}_MAE', round(iter_metrics.mae(),3))
        self.log(f'{stage}_MSE', round(iter_metrics.mse(),3))
        self.log(f'{stage}_RMSE', round(iter_metrics.rmse(),3))


    def on_test_epoch_end(self,):
        stage = "test"
        iter_metrics = self.metrics[stage]
        iter_metrics.aggregate() # aggregate results
        self.log(f'{stage}_recall', round(iter_metrics.recall(),3))
        self.log(f'{stage}_precision', round(iter_metrics.precision(),3))
        self.log(f'{stage}_f1-score', round(iter_metrics.fbeta_score(),3))
        self.log(f'{stage}_MAE', round(iter_metrics.mae(),3))
        self.log(f'{stage}_MSE', round(iter_metrics.mse(),3))
        self.log(f'{stage}_RMSE', round(iter_metrics.rmse(),3))
        
    
    def on_validation_epoch_start(self,):
        self.metrics["val"].flush()
    

    def on_test_epoch_start(self,):
        self.metrics["test"].flush()


    def training_step(self, batch, batch_idx):
        loss = self.shared_step("train", batch, batch_idx)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step("val", batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step("test",batch,batch_idx)
        return loss

    def configure_optimizers(self):


        optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                     lr=self.args.lr0, 
                                     weight_decay=self.args.weight_decay)


        return optimizer