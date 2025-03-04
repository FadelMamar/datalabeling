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
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from ..arguments import Arguments
from animaloc.models import HerdNet, load_model
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from animaloc.eval.lmds import HerdNetLMDS
from torch.nn import CrossEntropyLoss
import numpy as np
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir
from PIL import Image


def check_label_format(loaded_df: pd.DataFrame) -> str:
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
        raise NotImplementedError(
            f"The number of features ({num_features}) in the label file is wrong. Check yolo or yolo-obb format from ultralytics.")


def get_groundtruth(yolo_images_dir: str,
                    save_path: str = None,
                    load_gt_csv: str = None,
                    empty_ratio: float | None = 0.,
                    empty_frac: float | None = None):

    if empty_frac is not None:
        assert empty_frac >= 0. and empty_frac <= 1., "should be between 0 and 1."
    if empty_ratio is not None:
        assert empty_ratio >= 0., "should be non-negative"

    if load_gt_csv is not None:
        return pd.read_csv(load_gt_csv)

    cols1 = ['id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    cols2 = ['id', 'x', 'y', 'w', 'h']

    if not Path(yolo_images_dir).exists():
        raise FileNotFoundError('Image Directory does not exist.')

    # Iterate through labels
    dfs = list()
    for image_path in tqdm(Path(yolo_images_dir).glob("*"), desc="Getting groundtruths"):
        # load labels
        label_path = Path(str(image_path).replace(
            'images', 'labels')).with_suffix(".txt")
        if label_path.exists():
            # non-empty image
            df = pd.read_csv(label_path, sep=' ', header=None)
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
        else:
            # empty image -> creating pseudo labels
            df = {'id': [-1], 'x': [0.], 'y': [0.], 'w': [0.], 'h': [0.]}
            df = pd.DataFrame.from_dict(df, orient='columns')
            img_width, img_height = 0., 0.

        label_format = check_label_format(df)
        if label_format == 'yolo':
            df.columns = cols2
            df.loc[:, 'x'] = df['x']*img_width
            df.loc[:, 'y'] = df['y']*img_height
            df.loc[:, 'w'] = df['w']*img_width
            df.loc[:, 'h'] = df['h']*img_height
        else:  # yolo-obb
            df.columns = cols1
            df['x'] = (df['x1'] + df['x2'])*img_width*0.5
            df['y'] = (df['y1'] + df['y4'])*img_height*0.5
            df['w'] = (df['x2'] - df['x1'])*img_width
            df['h'] = (df['y4'] - df['y1'])*img_height

        df['images'] = str(image_path)
        df.rename(columns={'id': 'labels'}, inplace=True)
        dfs.append(df)

    # concat dfs
    dfs = pd.concat(dfs)
    # fix label range
    assert dfs["labels"].min() >= -1, "Check yolo label format."
    # shift to range [0,num_classes] so that 0 is the background class for empty images
    dfs["labels"] = dfs["labels"] + 1

    # sample empty images and non-empty
    # if there are
    if sum(dfs.labels < 1) > 0:
        df_non_empty = dfs.loc[dfs.labels > 0].copy()
        df_empty = dfs.loc[dfs.labels < 1].copy()
        if empty_frac is None:
            num_non_empty = len(df_non_empty)
            frac = min(empty_ratio*num_non_empty,
                       len(df_empty)
                       )/len(df_empty)
            frac = max(0., frac)
        else:
            frac = empty_frac
        df_empty = df_empty.sample(frac=frac)
        print(f"Sampling {len(df_empty)} empty images.", end="\n")
        # concatenate empty and non_empty
        dfs = pd.concat([df_empty, df_non_empty])

    if save_path is not None:
        dfs.to_csv(save_path, index=False)

    return dfs


def load_dataset(data_config_yaml: str,
                 split: str,
                 transforms: dict,
                 empty_ratio: float = 0.,
                 empty_frac: float = None):

    with open(data_config_yaml, 'r') as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    datasets = list()
    df_gts = list()
    root = data_config['path']
    for img_dir in tqdm(data_config[split], desc="concatenating datasets"):
        img_dir = os.path.join(root, img_dir)
        df = get_groundtruth(yolo_images_dir=img_dir,
                             save_path=None,
                             load_gt_csv=None,  # path_to_csv
                             empty_ratio=empty_ratio,
                             empty_frac=empty_frac
                             )
        dataset = CSVDataset(
            csv_file=df,
            root_dir=img_dir,
            albu_transforms=transforms[split][0],
            end_transforms=transforms[split][1]
        )
        datasets.append(dataset)
        df_gts.append(df)

    return ConcatDataset(datasets=datasets), pd.concat(df_gts)


class PredictDataset(CSVDataset):

    def __init__(self, images_path:list[str], albu_transforms = None, end_transforms = None):

        assert isinstance(images_path,list)

        images_path = list(map(str,images_path))
        # create dummy df_labels
        num_images = len(images_path)
        df_labels = {'x':[0.]*num_images,
                     'y':[0.]*num_images,
                     'labels':[0]*num_images,
                     'images':images_path
                     }
        df_labels = pd.DataFrame.from_dict(df_labels)
        super().__init__(csv_file=df_labels, 
                         root_dir="", 
                         albu_transforms=albu_transforms,
                         end_transforms=end_transforms)
    
    def _load_image(self, index: int) -> Image.Image:
        img_name = self._img_names[index]
        return Image.open(img_name).convert('RGB')
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:        
        img = self._load_image(index)
        target = self._load_target(index)
        tr_img, tr_target = self._transforms(img, target)

        return tr_img

class HerdnetData(L.LightningDataModule):

    def __init__(self, data_config_yaml: str,
                 patch_size: int,
                 down_ratio: int = 2,
                 batch_size: int = 32,
                 transforms: dict[str, tuple] = None,
                 train_empty_ratio: float = 0.):

        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.data_config_yaml = data_config_yaml
        self.transforms = transforms
        self.train_empty_ratio = train_empty_ratio
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.predict_dataset = None
        self.predict_batchsize = 8
        self.df_train_labels_freq = None
        self.df_val_labels_freq = None

        self.num_workers = 8
        self.pin_memory = torch.cuda.is_available()

        # Get number of classes
        with open(data_config_yaml, 'r') as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
            # accounting for background class
            self.num_classes = data_config['nc'] + 1

        if self.transforms is None:
            self.transforms = {}
            self.transforms['train'] = ([A.Resize(width=self.patch_size, height=self.patch_size, p=1.),
                                         A.VerticalFlip(p=0.5),
                                        A.HorizontalFlip(p=0.5),
                                        A.RandomRotate90(p=0.5),
                                        A.RandomBrightnessContrast(
                                            brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                                        A.Blur(blur_limit=15, p=0.2),
                                        A.Normalize(p=1.0, mean=(
                                            0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                         ],
                                        [
                                        MultiTransformsWrapper([
                                            FIDT(num_classes=self.num_classes,
                                                 down_ratio=down_ratio),
                                            PointsToMask(radius=2,
                                                         num_classes=self.num_classes,
                                                         squeeze=True,
                                                         down_ratio=int(
                                                             patch_size//(16*patch_size/512))
                                                         )
                                        ])]
                                        )
            self.transforms['val'] = (
                [A.Resize(width=self.patch_size, height=self.patch_size, p=1.),
                 A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
                 ],
                [
                    DownSample(
                        down_ratio=down_ratio, anno_type='point'),
                ]
            )
            self.transforms['test'] = self.transforms['val']

    @property
    def get_labels_weights(self,):
        # if self.num_classes == 2:
        #     return [1.0, 1.0]
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
                                                               empty_ratio=self.train_empty_ratio,
                                                               empty_frac=None
                                                               )
            self.df_train_labels_freq = df_train_labels['labels'].value_counts(
            ).sort_index()/len(df_train_labels)
            # val
            self.val_dataset, df_val_labels = load_dataset(data_config_yaml=self.data_config_yaml,
                                                           split='val',
                                                           transforms=self.transforms,
                                                           empty_ratio=None,
                                                           empty_frac=1.
                                                           )
            self.df_val_labels_freq = df_val_labels['labels'].value_counts(
            ).sort_index()/len(df_val_labels)

        elif stage == "test":
            self.test_dataset, _ = load_dataset(data_config_yaml=self.data_config_yaml,
                                                split='test',
                                                transforms=self.transforms,
                                                empty_frac=1.,
                                                empty_ratio=None
                                                )

    def set_predict_dataset(self,images_path:list[str],batchsize:int=16):        
        self.predict_dataset = PredictDataset(images_path=images_path,
                                              albu_transforms=self.transforms['val'][0],
                                              end_transforms=self.transforms['val'][1]
                                            )
        self.predict_batchsize = batchsize

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, collate_fn=None,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=True
                        )

    def val_dataloader(self):
        """Validation dataloader supports only batchsize=1.


        Returns
        -------
        DataLoader
            validation DataLoader.

        """
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=None)

    def test_dataloader(self):
        """Test dataloader supports only batchsize=1.


        Returns
        -------
        DataLoader
            test DataLoader.

        """
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=None)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.predict_batchsize, shuffle=False)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


class HerdnetTrainer(L.LightningModule):

    def __init__(self,
                 herdnet_model_path: str,
                 args: Arguments,
                 work_dir: str,
                 eval_radius:int=20,
                 load_state_dict_strict: bool = True,
                 ce_weight: list = None):

        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.work_dir = work_dir

        # Get number of classes
        with open(self.args.data_config_yaml, 'r') as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
            # including a class for background
            num_classes = data_config['nc'] + 1
        self.class_mapping = {str(k+1):v for k,v in data_config['names'].items()}

        ce_weight = torch.Tensor(ce_weight) if (
            ce_weight is not None) else None
        losses = [
            {'loss': FocalLoss(reduction='mean'), 'idx': 0,
             'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
            {'loss': CrossEntropyLoss(
                reduction='mean', weight=ce_weight), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
        ]

        self.model = HerdNet(pretrained=False, down_ratio=2, num_classes=4)
        self.model = LossWrapper(self.model, losses=losses, mode="both")
        checkpoint = torch.load(
            herdnet_model_path, map_location="cpu", weights_only=True)
        success = self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=load_state_dict_strict)
        self.model.model.reshape_classes(num_classes)
        print(f"Loading ckpt:", success)

        # metrics
        self.metrics_val = PointsMetrics(
            radius=eval_radius, num_classes=num_classes)
        self.metrics_test = PointsMetrics(
            radius=eval_radius, num_classes=num_classes)

        self.metrics = {"val": self.metrics_val, "test": self.metrics_test}

        self.stitcher = None
        # self.stitcher = HerdNetStitcher(
        #                 model=herdnet,
        #                 size=(self.args.imgsz,self.args.imgsz),
        #                 overlap=160,
        #                 down_ratio=down_ratio,
        #                 reduction='mean'
        #                 )

        self.herdnet_evaluator = HerdNetEvaluator(
            model=self.model,
            dataloader=DataLoader(dataset=[None, None], batch_size=1),
            metrics=self.metrics_val,
            stitcher=self.stitcher,
            work_dir=self.work_dir,
            header='validation'
        )
        up = True
        if self.stitcher is not None:
            up = False
        self.lmds = HerdNetLMDS(up=up, **self.herdnet_evaluator.lmds_kwargs)

    # def configure_model(self,):
    #     self.model = torch.compile(self.model, fullgraph=True)

    def prepare_feeding(self, targets: dict[str, torch.Tensor] | None, output: dict[torch.Tensor]) -> dict:
        # copy and adapted from animaloc.eval.HerdnetEvaluator

        gt = dict(loc=[[None, None]], labels=[None])
        if targets is not None:
            gt_coords = [p[::-1]
                         for p in targets['points'].squeeze(0).tolist()]
            gt_labels = targets['labels'].squeeze(0).tolist()

            gt = dict(
                loc=gt_coords,
                labels=gt_labels
            )

        

        counts, locs, labels, scores, dscores = self.lmds(output)

        preds = dict(
            loc=locs[0],
            labels=labels[0],
            scores=scores[0],
            dscores=dscores[0]
        )

        return dict(gt=gt, preds=preds, est_count=counts[0])

    def shared_step(self, stage, batch, batch_idx):

        images, targets = batch

        # compute losses
        if stage == "train":
            predictions, loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            self.log_dict(loss_dict)
            return loss

        else:
            predictions, _ = self.model(images)

            # compute metrics
            output = self.prepare_feeding(
                targets=targets, output=predictions)
            if output['gt']['labels'][0] < 1:  # labels = 0 -> background
                output = {'gt': {'loc': [], 'labels': []},
                          'preds': output['preds'],
                          'est_count': output['est_count']
                          }
            iter_metrics = self.metrics[stage]
            iter_metrics.feed(**output)
            return None

    def log_metrics(self, stage: str):

        iter_metrics = self.metrics[stage]

        # store for class level metrics computation
        self.herdnet_evaluator._stored_metrics = iter_metrics.copy()

        # aggregate results
        iter_metrics.aggregate()
        self.log(f'{stage}_recall', round(iter_metrics.recall(), 3))
        self.log(f'{stage}_precision', round(iter_metrics.precision(), 3))
        self.log(f'{stage}_f1-score', round(iter_metrics.fbeta_score(), 3))
        self.log(f'{stage}_MAE', round(iter_metrics.mae(), 3))
        self.log(f'{stage}_MSE', round(iter_metrics.mse(), 3))
        self.log(f'{stage}_RMSE', round(iter_metrics.rmse(), 3))

        # log perclass metrics
        per_class_metrics = self.herdnet_evaluator.results
        metrics_cols = [
            p for p in per_class_metrics.columns if p not in ['class',]]
        for _,row in per_class_metrics.iterrows():
            for col in metrics_cols:
                label = str(row.loc['class'])
                class_name =self.class_mapping[label]
                self.log(f"{class_name}_{col}", round(row.loc[col], 3))

    def on_validation_epoch_end(self,):
        self.log_metrics(stage='val')

    def on_test_epoch_end(self,):
        self.log_metrics(stage='test')

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
        loss = self.shared_step("test", batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):

        images = batch
        predictions, _ = self.model(images)

        # compute metrics
        output = self.prepare_feeding(targets=None,
                                      output=predictions)
        output.pop('gt') # empty
        return output

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.args.lr0,
                                     weight_decay=self.args.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=self.args.epochs,
                                                                            T_mult=1,
                                                                            eta_min=self.args.lr0*self.args.lrf,
                                                                        )
        return [optimizer],  [{"scheduler": lr_scheduler, "interval": "epoch"}]
     
        

        
