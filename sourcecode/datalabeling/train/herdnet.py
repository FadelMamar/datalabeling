import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
import albumentations as A
import yaml
from animaloc.datasets import CSVDataset, FolderDataset
from animaloc.data.transforms import (
    MultiTransformsWrapper,
    DownSample,
    PointsToMask,
    FIDT,
)
import pandas as pd
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from animaloc.models import HerdNet
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from animaloc.eval.lmds import HerdNetLMDS
from torch.nn import CrossEntropyLoss
from animaloc.eval import PointsMetrics, HerdNetEvaluator
from PIL import Image
from ..dataset.converters import check_label_format


def get_groundtruth(
    yolo_images_dir: str,
    save_path: str = None,
    load_gt_csv: str = None,
) -> tuple[pd.DataFrame, int, list]:
    if load_gt_csv is not None:
        return pd.read_csv(load_gt_csv)

    cols1 = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    cols2 = ["id", "x", "y", "w", "h"]

    if not Path(yolo_images_dir).exists():
        raise FileNotFoundError("Image Directory does not exist.")

    # Iterate through labels
    dfs = list()
    count_empty = 0
    images_empty = list()
    for image_path in tqdm(
        Path(yolo_images_dir).glob("*"), desc="Getting groundtruths"
    ):
        # load labels
        label_path = Path(str(image_path).replace("images", "labels")).with_suffix(
            ".txt"
        )
        if label_path.exists():
            # non-empty image
            df = pd.read_csv(label_path, sep=" ", header=None)
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
        else:  # empty image has no corresponding txt file
            count_empty += 1
            images_empty.append(str(image_path))
            continue
            # empty image -> creating pseudo labels
            # df = {'id': [-1], 'x': [0.], 'y': [0.], 'w': [0.], 'h': [0.]}
            # df = pd.DataFrame.from_dict(df, orient='columns')
            # img_width, img_height = 0., 0.

        label_format = check_label_format(df)
        if label_format == "yolo":
            df.columns = cols2
            df.loc[:, "x"] = df["x"] * img_width
            df.loc[:, "y"] = df["y"] * img_height
            df.loc[:, "w"] = df["w"] * img_width
            df.loc[:, "h"] = df["h"] * img_height
        else:  # yolo-obb
            df.columns = cols1
            df["x"] = (df["x1"] + df["x2"]) * img_width * 0.5
            df["y"] = (df["y1"] + df["y4"]) * img_height * 0.5
            df["w"] = (df["x2"] - df["x1"]) * img_width
            df["h"] = (df["y4"] - df["y1"]) * img_height
            df.drop(columns=cols1[1:],inplace=True)

        df["images"] = str(image_path)
        df.rename(columns={"id": "labels"}, inplace=True)
        dfs.append(df)

    # concat dfs
    dfs = pd.concat(dfs)
    # fix label range
    assert dfs["labels"].min() >= 0, "Check yolo label format."
    # shift to range [1,num_classes] so that 0 is the background class for empty images
    dfs["labels"] = dfs["labels"] + 1

    if save_path is not None:
        dfs.to_csv(save_path, index=False)

    return dfs, count_empty, images_empty


def load_dataset(
    data_config_yaml: str,
    split: str,
    transforms: dict,
    empty_ratio: float | None = None,
    empty_frac: float | None = None,
) -> tuple[ConcatDataset, pd.DataFrame, int]:
    """Load dataset from yaml file.

    Args:
        data_config_yaml (str): data configuration yaml file.
        split (str): train, val or test.
        transforms (dict): transforms for the dataset.
        empty_ratio (float | None, optional): samples empty images following empty_ratio*num_positive_samples. Defaults to None.
        empty_frac (float | None, optional): samples empty images following pd.Series(list_empty_images).sample(frac=empty_frac). Defaults to None.

    Returns:
        tuple[ConcatDataset, pd.DataFrame, int]: _description_
    """

    assert split in ["train", "val", "test"], "split should be either train, val or test."
    if empty_frac is not None:
        assert empty_frac >= 0.0 and empty_frac <= 1.0, "should be between 0 and 1."
        assert empty_ratio is None, "empty_ratio should be None if empty_frac is not None."
    if empty_ratio is not None:
        assert empty_ratio >= 0.0, "should be non-negative"
        assert empty_frac is None, "empty_frac should be None if empty_ratio is not None."

    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    datasets = list()
    df_gts = list()
    root = data_config["path"]
    num_empty_images = 0
    for img_dir in tqdm(data_config[split], desc="concatenating datasets"):
        img_dir = os.path.join(root, img_dir)
        df, count_empty, images_empty = get_groundtruth(
            yolo_images_dir=img_dir,
            save_path=None,
            load_gt_csv=None,
        )
        # select empty images
        num_empty_sampled = None
        if empty_ratio is not None:
            num_empty_sampled = (
                min(int(empty_ratio * len(df)), len(images_empty))
                if empty_frac is not None
                else None
            )
            # print('num_empty_sampled: ',num_empty_sampled)
        if (num_empty_sampled is None) and (empty_frac is None):
            sampled_images_empty = []
        else:
            sampled_images_empty = (
                pd.Series(images_empty)
                .sample(
                    n=num_empty_sampled, frac=empty_frac, replace=False, random_state=41
                )
                .to_list()
            )
            # df_empty = pd.DataFrame()
            # df_empty['images'] = sampled_images_empty
            # for col in ["labels", "x", "y", "w", "h"]:
            #     df_empty[col] = 0
            # df = pd.concat([df, df_empty])
        
        # selected images
        selected_images = sampled_images_empty + df["images"].to_list()
        selected_images = list(set(selected_images))
        dataset = FolderDataset(  # CSVDataset FolderDataset
            csv_file=df,
            root_dir="",
            albu_transforms=transforms[split][0],
            end_transforms=transforms[split][1],
            images_paths=selected_images, # FolderDataset
        )

        datasets.append(dataset)
        df_gts.append(df)
        
        num_empty_images += len(sampled_images_empty)

    return ConcatDataset(datasets=datasets), pd.concat(df_gts), num_empty_images


class PredictDataset(CSVDataset):
    def __init__(
        self, images_path: list[str], albu_transforms=None, end_transforms=None
    ):
        assert isinstance(images_path, list)

        images_path = list(map(str, images_path))
        # create dummy df_labels
        num_images = len(images_path)
        df_labels = {
            "x": [0.0] * num_images,
            "y": [0.0] * num_images,
            "labels": [0] * num_images,
            "images": images_path,
        }
        df_labels = pd.DataFrame.from_dict(df_labels)
        super().__init__(
            csv_file=df_labels,
            root_dir="",
            albu_transforms=albu_transforms,
            end_transforms=end_transforms,
        )

    def _load_image(self, index: int) -> Image.Image:
        img_name = self._img_names[index]
        return Image.open(img_name).convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        img = self._load_image(index)
        target = self._load_target(index)
        tr_img, tr_target = self._transforms(img, target)

        return tr_img


class HerdnetData(L.LightningDataModule):
    """Lightning datamodule. This class handles all the data preparation tasks. It facilitates reproducibility."""

    def __init__(
        self,
        data_config_yaml: str,
        patch_size: int,
        down_ratio: int = 2,
        batch_size: int = 32,
        transforms: dict[str, tuple] = None,
        train_empty_ratio: float = 0.0,
        normalization: str = "standard",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
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
        self.num_empty_images_val = None
        self.num_empty_images_train = None
        self.num_empty_images_test = None

        self.num_workers = 8
        self.pin_memory = torch.cuda.is_available()

        # Get number of classes
        with open(data_config_yaml, "r") as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
            # accounting for background class
            self.num_classes = data_config["nc"] + 1

        if self.transforms is None:
            self.transforms = {}
            self.transforms["train"] = (
                [
                    A.Resize(width=self.patch_size, height=self.patch_size, p=1.0),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.2
                    ),
                    A.Blur(blur_limit=15, p=0.2),
                    A.Normalize(
                        normalization=normalization,
                        p=1.0,
                        mean=mean,
                        std=std,
                    ),
                ],
                [
                    MultiTransformsWrapper(
                        [
                            FIDT(num_classes=self.num_classes, down_ratio=down_ratio),
                            PointsToMask(
                                radius=2,
                                num_classes=self.num_classes,
                                squeeze=True,
                                down_ratio=int(patch_size // (16 * patch_size / 512)),
                            ),
                        ]
                    )
                ],
            )
            self.transforms["val"] = (
                [
                    A.Resize(width=self.patch_size, height=self.patch_size, p=1.0),
                    A.Normalize(
                        normalization=normalization,
                        p=1.0,
                        mean=mean,
                        std=std,
                    ),
                ],
                [
                    DownSample(down_ratio=down_ratio, anno_type="point"),
                ],
            )
            self.transforms["test"] = self.transforms["val"]

    @property
    def get_labels_weights(
        self,
    ) -> torch.Tensor:
        """Computes importance weights for cross entropy loss

        Returns:
            torch.Tensor: weights for cross entropy loss
        """
        weights = 1 / (self.df_train_labels_freq + 1e-6)
        weights = [1.0] + weights.to_list()
        assert len(weights) == self.num_classes, "Check for inconsistencies."
        return torch.Tensor(weights)

    def setup(self, stage: str):
        if stage == "fit":
            # train
            self.train_dataset, df_train_labels, self.num_empty_images_train = (
                load_dataset(
                    data_config_yaml=self.data_config_yaml,
                    split="train",
                    transforms=self.transforms,
                    empty_ratio=self.train_empty_ratio,
                    empty_frac=None,
                )
            )
            self.df_train_labels_freq = df_train_labels[
                "labels"
            ].value_counts().sort_index() / (
                len(df_train_labels) + self.num_empty_images_train
            )
            print(
                f"Train dataset as {len(self.train_dataset)} samples including {self.num_empty_images_train} negative samples."
            )
            # val
            self.val_dataset, df_val_labels, self.num_empty_images_val = load_dataset(
                data_config_yaml=self.data_config_yaml,
                split="val",
                transforms=self.transforms,
                empty_ratio=None,
                empty_frac=1.0,
            )
            self.df_val_labels_freq = df_val_labels[
                "labels"
            ].value_counts().sort_index() / (
                len(df_val_labels) + self.num_empty_images_val
            )
            print(
                f"Val dataset as {len(self.val_dataset)} samples including {self.num_empty_images_val} negative samples."
            )

        elif stage == "test":
            self.test_dataset, _, self.num_empty_images_test = load_dataset(
                data_config_yaml=self.data_config_yaml,
                split="test",
                transforms=self.transforms,
                empty_frac=1.0,
                empty_ratio=None,
            )
            print(
                f"Test dataset as {len(self.test_dataset)} samples including {self.num_empty_images_test} negative samples."
            )
        elif stage == "validate":
            # val
            self.val_dataset, df_val_labels, self.num_empty_images_val = load_dataset(
                data_config_yaml=self.data_config_yaml,
                split="val",
                transforms=self.transforms,
                empty_ratio=None,
                empty_frac=1.0,
            )
            self.df_val_labels_freq = df_val_labels[
                "labels"
            ].value_counts().sort_index() / (
                len(df_val_labels) + self.num_empty_images_val
            )
            print(
                f"Val dataset as {len(self.val_dataset)} samples including {self.num_empty_images_val} negative samples."
            )

    def val_collate_fn(self, batch: tuple) -> tuple[torch.Tensor, dict]:
        """collate_fn used to create the validation dataloader

        Args:
            batch (tuple): (img:torch.Tensor, targets:dict)

        Returns:
            tuple: (image, target)
        """

        batched = dict(points=[], labels=[])
        batch_img = torch.stack([p[0] for p in batch])
        targets = [p[1] for p in batch]
        keys = targets[0].keys()

        # get non_empty samples indidces -> set difference
        non_empty_idx = [i for i, a in enumerate(targets) if len(a["labels"]) > 0]
        targets_empty = [
            targets[i] for i in list(set(range(len(batch))) - set(non_empty_idx))
        ]
        targets = [targets[i] for i in non_empty_idx]

        # Creating batch
        for k in keys:
            batched[k] = []  # initialize to be empty list
            if k == "points" or k=='labels':
                batched[k] = [a[k].cpu().tolist() for a in targets]
                if len(targets_empty) > 0:
                    batched[k] = batched[k] + [[]] * len(targets_empty)
            # if k == "labels":
            #     batched[k] = [a[k].cpu().tolist() for a in targets]
            #     # batched[k] = [a if isinstance(a,list) else [a] for a in batched[k]]
            #     if len(targets_empty) > 0:
            #         batched[k] = batched[k] + [[]] * len(targets_empty)

        return batch_img, batched

    def set_predict_dataset(self, images_path: list[str], batchsize: int = 16) -> None:
        self.predict_dataset = PredictDataset(
            images_path=images_path,
            albu_transforms=self.transforms["val"][0],
            end_transforms=self.transforms["val"][1],
        )
        self.predict_batchsize = batchsize

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Creates validation dataloader.


        Returns
        -------
        DataLoader
            validation DataLoader.

        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=torch.utils.data.SequentialSampler(self.val_dataset),
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            persistent_workers=True
        )

    def test_dataloader(self):
        """Test dataloader .


        Returns
        -------
        DataLoader
            test DataLoader.

        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            shuffle=False,
            # num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            # persistent_workers=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.predict_batchsize, shuffle=False
        )


class HerdnetTrainer(L.LightningModule):
    def __init__(
        self,
        data_config_yaml: str,
        lr: float,
        weight_decay: float,
        loaded_weights_num_classes: int,
        work_dir: str,
        eval_radius: int = 20,
        down_ratio: int = 2,
        classification_threshold: float = 0.25,
        load_state_dict_strict: bool = True,
        herdnet_model_path: str | None = None,
        ce_weight: torch.Tensor = None,
        losses: list = None,
        epochs: int=None,
        lrf:float=1e-1
    ):
        super().__init__()

        self.save_hyperparameters(
            "lr",
            "weight_decay",
            "data_config_yaml",
            "down_ratio",
            "ce_weight",
            "eval_radius",
            "lrf",
            "epochs"
        )

        self.work_dir = work_dir
        self.loaded_weights_num_classes = loaded_weights_num_classes
        self.classification_threshold = classification_threshold

        # Get number of classes
        with open(data_config_yaml, "r") as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
            # including a class for background
            self.num_classes = data_config["nc"] + 1
        self.class_mapping = {str(k + 1): v for k, v in data_config["names"].items()}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if losses is None:
            losses = [
                {
                    "loss": FocalLoss(reduction="mean"),
                    "idx": 0,
                    "idy": 0,
                    "lambda": 1.0,
                    "name": "focal_loss",
                },
                {
                    "loss": CrossEntropyLoss(reduction="mean", weight=ce_weight),
                    "idx": 1,
                    "idy": 1,
                    "lambda": 1.0,
                    "name": "ce_loss",
                },
            ]
        # Load herdnet object
        self.model = HerdNet(
            pretrained=False,
            down_ratio=down_ratio,
            num_classes=loaded_weights_num_classes,
        )
        self.model = LossWrapper(self.model, losses=losses, mode="both")
        if herdnet_model_path is not None:
            try:
                checkpoint = torch.load(
                    herdnet_model_path, map_location=device, weights_only=True
                )
                success = self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=load_state_dict_strict
                )
                print("Loading ckpt:", herdnet_model_path)
            except:
                checkpoint = torch.load(
                    herdnet_model_path, map_location=device, weights_only=True
                )
                success = self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                print("Warning! load_state_dict_strict is being set to False")
                print(success)
                

        if self.num_classes != self.loaded_weights_num_classes:
            print(
                f"Classification head of herdnet will be modified to handle {self.num_classes} classes."
            )
            self.model.model.reshape_classes(self.num_classes)
            

        # metrics
        self.metrics_val = PointsMetrics(
            radius=eval_radius, num_classes=self.num_classes
        )
        self.metrics_test = PointsMetrics(
            radius=eval_radius, num_classes=self.num_classes
        )

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
            header="validation",
            lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 3.0, "neg_ts": 0.1},
        )
        up = True
        if self.stitcher is not None:
            up = False
        self.lmds = HerdNetLMDS(up=up, **self.herdnet_evaluator.lmds_kwargs)

    def batch_metrics(
        self, metric: PointsMetrics, batchsize: int, output: dict
    ) -> None:
        if batchsize >= 1:
            for i in range(batchsize):
                gt = {k: v[i] for k, v in output["gt"].items()}
                preds = {k: v[i] for k, v in output["preds"].items()}
                counts = output["est_count"][i]
                output_i = dict(gt=gt, preds=preds, est_count=counts)
                metric.feed(**output_i)
        else:
            raise NotImplementedError

    def prepare_feeding(
        self, targets: dict[str, torch.Tensor], output: list[torch.Tensor]
    ) -> dict:
        try:  # batchsize==1
            gt_coords = [p[::-1] for p in targets["points"].cpu().tolist()]
            gt_labels = targets["labels"].cpu().tolist()
        except Exception:  # batchsize>1
            gt_coords = [p[::-1] for p in targets["points"]]
            gt_labels = targets["labels"]

        # get predictions
        counts, locs, labels, scores, dscores = self.lmds(output)
        gt = dict(loc=gt_coords, labels=gt_labels)
        preds = dict(loc=locs, labels=labels, scores=scores, dscores=dscores)

        return dict(gt=gt, preds=preds, est_count=counts)

    def shared_step(self, stage, batch, batch_idx):
        

        # compute losses
        if stage == "train":
            images, targets = batch
            predictions, loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            self.log_dict(loss_dict)
            return loss.clamp(-5., 5.) # preventing exploding gradient

        else:
            images, targets = batch
            batchsize = images.shape[0]
            assert batchsize >= 1 and len(images.shape) == 4, (
                "Input image does not have the right shape > e.g. [b,c,h,w]"
            )
            predictions, _ = self.model(images)
            # compute metrics
            output = self.prepare_feeding(targets=targets, output=predictions)
            iter_metrics = self.metrics[stage]
            self.batch_metrics(metric=iter_metrics, batchsize=batchsize, output=output)
            return None

    def log_metrics(self, stage: str):
        assert stage != "train", "metrics only logged for val and test."

        iter_metrics = self.metrics[stage]

        # store for class level metrics computation
        self.herdnet_evaluator._stored_metrics = iter_metrics.copy()

        # aggregate results
        iter_metrics.aggregate()
        self.log(f"{stage}_recall", round(iter_metrics.recall(), 3))
        self.log(f"{stage}_precision", round(iter_metrics.precision(), 3))
        self.log(f"{stage}_f1-score", round(iter_metrics.fbeta_score(), 3))
        self.log(f"{stage}_MAE", round(iter_metrics.mae(), 3))
        self.log(f"{stage}_MSE", round(iter_metrics.mse(), 3))
        self.log(f"{stage}_RMSE", round(iter_metrics.rmse(), 3))

        # log perclass metrics
        per_class_metrics = self.herdnet_evaluator.results
        metrics_cols = [
            p
            for p in per_class_metrics.columns
            if p
            not in [
                "class",
            ]
        ]
        for _, row in per_class_metrics.iterrows():
            for col in metrics_cols:
                label = str(row.loc["class"])
                if label in self.class_mapping.keys():
                    class_name = self.class_mapping[label]
                    name = f"{class_name}_{col}"
                else:
                    name = label
                self.log(name, round(row.loc[col], 3))

    def on_validation_epoch_end(
        self,
    ):
        self.log_metrics(stage="val")

    def on_test_epoch_end(
        self,
    ):
        self.log_metrics(stage="test")

    def on_validation_epoch_start(
        self,
    ):
        self.metrics["val"].flush()

    def on_test_epoch_start(
        self,
    ):
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
        output = self.prepare_feeding(targets=None, output=predictions)
        output.pop("gt")  # empty
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=self.hparams.epochs,
                                                                            T_mult=1,
                                                                            eta_min=self.hparams.lr*self.hparams.lrf,
                                                                        )
        return [optimizer],  [{"scheduler": lr_scheduler, "interval": "epoch"}]
        # return optimizer
