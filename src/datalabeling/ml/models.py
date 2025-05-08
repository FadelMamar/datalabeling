import logging
import traceback
from pathlib import Path

import geopy
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from torch.utils.data import DataLoader, TensorDataset
import requests, base64

# from label_studio_ml.utils import (get_env, get_local_path)
from tqdm import tqdm
from ultralytics import YOLO

from ..common.annotation_utils import GPSUtils, ImageProcessor

logger = logging.getLogger(__name__)


class Detector(object):
    def __init__(
        self,
        path_to_weights: str,
        confidence_threshold: float = 0.1,
        overlap_ratio: float = 0.1,
        tilesize: int | None = 1280,
        imgsz: int = 1280,
        device: str = None,
        use_sliding_window: bool = True,
        is_yolo_obb: bool = False,
    ):
        """_summary_

        Args:
            path_to_weights (str): _description_
            confidence_threshold (float, optional): _description_. Defaults to 0.1.
            overlap_ratio (float, optional): _description_. Defaults to 0.1.
            tilesize (int | None, optional): _description_. Defaults to 1280.
            imgsz (int, optional): _description_. Defaults to 1280.
            device (str, optional): _description_. Defaults to None.
            use_sliding_window (bool, optional): _description_. Defaults to True.
            is_yolo_obb (bool, optional): _description_. Defaults to False.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tilesize = tilesize
        self.imgsz = imgsz
        self.overlapratio = overlap_ratio
        self.use_sliding_window = use_sliding_window
        self.is_yolo_obb = is_yolo_obb

        self.detection_model = None
        if path_to_weights is not None:
            logger.info(f"Computing device: {device}")

            self.detection_model = UltralyticsDetectionModel(
                model=YOLO(path_to_weights, task="detect"),
                confidence_threshold=confidence_threshold,
                image_size=self.imgsz,
                device=device,
            )

    def _predict_url(
        self,
        image_path: str,
        inference_service_url: str = "http://127.0.0.1:4141/predict",
        timeout=3 * 60,
        return_gps: bool = True,
    ):
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "image": img_b64,
            "sahi_prostprocess": "NMS",
            "override_tilesize": None,  # tilesize to use for
            "postprocess_match_threshold": 0.5,
            "nms_iou": None,
            "return_as_decimal": False,
        }

        resp = requests.post(
            inference_service_url,
            json=payload,
            timeout=timeout,
        ).json()

        detections = resp["detections"]
        image_gps = resp["image_gps"]

        if return_gps:
            return detections, image_gps

        return detections

    # TODO: batch predictions with slicing
    def predict(
        self,
        image: Image.Image = None,
        inference_service_url: str = None,
        image_path: str = None,
        return_gps: bool = False,
        return_coco: bool = False,
        sahi_prostprocess: float = "NMS",
        override_tilesize: int = None,
        postprocess_match_threshold: float = 0.5,
        timeout: int = 3 * 60,
        nms_iou: float = None,
        verbose: int = 0,
    ):
        """Run sliced predictions

        Args:
            image (Image): input image

        Returns:
            dict: predictions in coco format
        """

        # predict using inference service
        if isinstance(inference_service_url, str):
            self._predict_url(
                image_path=image_path,
                inference_service_url=inference_service_url,
                timeout=timeout,
                return_gps=return_gps,
            )

        # predict using local model
        if image is None:
            assert image_path is not None, "Provide the image path."
            image = Image.open(image_path)
        else:
            assert isinstance(image, Image.Image)

        if self.use_sliding_window:
            tilesize = override_tilesize or self.tilesize
            result = get_sliced_prediction(
                image,
                self.detection_model,
                slice_height=tilesize,
                slice_width=tilesize,
                overlap_height_ratio=self.overlapratio,
                overlap_width_ratio=self.overlapratio,
                postprocess_type=sahi_prostprocess,
                postprocess_match_metric="IOU",
                verbose=verbose,
                postprocess_match_threshold=postprocess_match_threshold or nms_iou,
            )
        else:
            result = get_prediction(
                image=image,
                detection_model=self.detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=verbose,
            )

        if return_coco:
            result = result.to_coco_annotations()

        out = result
        # get gps coordinates
        if return_gps:
            gps_coords = GPSUtils.get_gps_coord(file_name=image_path, image=image)
            out = result, gps_coords

        return out

    # TODO: to debug and optimize
    def sliced_prediction(
        self,
        image_path: str,
        image: Image = None,
        patchsize: int = 640,
        stride: int = 128,
    ):
        if image is None:
            assert image_path is not None, "Provide the image path."
            image = Image.open(image_path)

        image = image.convert("RGB")
        to_tensor = T.ToTensor()
        image = to_tensor(image).unsqueeze(0)
        image = image[:, ::-1, :, :]

        # unfold gives shape [1, C*ph*pw, L] where L = number of patches
        patches_flat = F.unfold(
            image, kernel_size=(patchsize, patchsize), stride=(stride, stride)
        )
        batch_of_patches = patches_flat.transpose(1, 2).reshape(
            -1, 3, patchsize, patchsize
        )

        # number of patches along width, height:
        H, W = image.shape[2:]
        n_w = (W - patchsize) // stride + 1
        n_h = (H - patchsize) // stride + 1

        # for patch index k in [0..L-1]:
        row_idx = lambda k: k // n_h
        col_idx = lambda k: k % n_w

        # top‐left pixel of this patch in original:
        y0 = lambda i: row_idx(i) * stride
        x0 = lambda j: col_idx(j) * stride

        dataset = TensorDataset(batch_of_patches)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        results = []
        indexes = []
        offset = 0
        with torch.no_grad():
            for (batch,) in loader:
                res = self.detection_model.model(batch)
                results.append(res)
                indexes = indexes + list(range(offset, offset + batch.shape[0]))
                offset += batch.shape[0]

        # TODO: debug top-left pixels in the original image
        y0_x0 = [(y0(i), x0(i)) for i in indexes]

        return results, y0_x0

    def predict_directory(
        self,
        path_to_dir: str = None,
        images_paths: list[str] = None,
        return_gps: bool = False,
        return_coco: bool = False,
        as_dataframe: bool = True,
        save_path: str = None,
    ) -> dict | pd.DataFrame:
        """Computes predictions on a directory

        Args:
            path_to_dir (str): path to directory with images. Defaults to None
            images_list (list): paths of images to run the detection on
            as_dataframe (bool): returns results as pd.DataFrame
            save_path (str) : converts to dataframe and then save

        Returns:
            dict: a directory with the schema {image_path:prediction_coco_format}
        """

        assert (path_to_dir is None) + (images_paths is None) < 2, (
            "Both should not be given."
        )
        results = {}
        paths = images_paths or list(Path(path_to_dir).iterdir())
        for image_path in tqdm(paths, desc="Computing predictions..."):

            try:
                pred = self.predict(
                    image=None,
                    return_coco=return_coco or as_dataframe or return_gps,
                    image_path=image_path,
                    return_gps=return_gps,
                )
            except Exception as e:
                logger.error(e)
                logger.error(f"Failed for {image_path}")
                continue
            gps_coords = None
            if return_gps:
                pred, gps_info = pred
                if isinstance(gps_info, tuple):
                    gps_coords = gps_info[0]

            pred.append(dict(gps_coords=gps_coords))
            results.update({str(image_path): pred})

        # returns as df or save
        if as_dataframe or (save_path is not None):
            results = self.get_pred_results_as_dataframe(results, return_gps=return_gps)

            if save_path is not None:
                try:
                    results.to_json(save_path, orient="records", indent=2)
                except Exception:
                    logger.info("!!!Failed to save results as json!!!\n")
                    traceback.print_exc()

        return results

    def format_gps(self, gps_coord: str):
        if gps_coord is None:
            return None, None, None

        point = geopy.Point.from_string(gps_coord)

        lat = point.latitude
        long = point.longitude
        alt = point.altitude * 1e3  # converting to meters

        return lat, long, alt

    def get_detections_gps(
        self,
        x: pd.Series,
        flight_height: int = 180,
        sensor_height: int = 24,
        gsd: float = None,
    ):
        try:
            img_path = x["file_name"]

            exif = GPSUtils.get_exif(img_path)
            if exif is None:
                return pd.Series(
                    data=[None, None],
                )
            H = exif["ExifImageHeight"]
            W = exif["ExifImageWidth"]

            lat_center, lon_center, alt = x.Latitude, x.Longitude, x.Elevation

            # bbox center
            x_det = x["x_min"] + x["bbox_w"] * 0.5
            y_det = x["y_min"] + x["bbox_h"] * 0.5

            if gsd is None:
                gsd = ImageProcessor.get_gsd(
                    image_path=img_path,
                    sensor_height=sensor_height,
                    flight_height=flight_height,
                )

            gsd *= 1e-2  # convert to m/px

            px_lat, px_long = ImageProcessor.generate_pixel_coordinates(
                x=x_det,
                y=y_det,
                lat_center=lat_center,
                lon_center=lon_center,
                W=W,
                H=H,
                gsd=gsd,
            )
            return pd.Series(
                data=[px_lat, px_long],
            )

        except Exception as e:
            traceback.print_exc()
            return pd.Series(
                data=[None, None],
            )

    def get_pred_results_as_dataframe(
        self, results: dict[str:list], return_gps: bool = False
    ):
        unravel_dict = []
        gps_coords = []
        for key, value in results.items():
            for v in value:
                if "gps_coords" not in v.keys():
                    unravel_dict.append(
                        {"file_name": key, "value": v}
                    )  # = v #,results['gps_coords']
                elif return_gps:
                    gps_coords.append({"gps": v["gps_coords"], "file_name": key})

        df_results = pd.DataFrame.from_dict(unravel_dict)

        dfs = list()
        for i in tqdm(range(len(df_results)), desc="pred results as df"):
            df_i = pd.DataFrame.from_records(df_results.iloc[i, 1:].to_list())
            df_i.loc[0, "file_name"] = df_results.iat[i, 0]

            dfs.append(df_i)

        dfs = pd.concat(dfs, axis=0).dropna(thresh=5)

        dfs["x_min"] = dfs["bbox"].apply(lambda x: x[0])
        dfs["y_min"] = dfs["bbox"].apply(lambda x: x[1])
        dfs["bbox_w"] = dfs["bbox"].apply(lambda x: x[2])
        dfs["bbox_h"] = dfs["bbox"].apply(lambda x: x[3])
        dfs["x_max"] = dfs["x_min"] + dfs["bbox_w"]
        dfs["y_max"] = dfs["y_min"] + dfs["bbox_h"]

        # add gps info
        if return_gps:
            df_gps = pd.DataFrame.from_dict(gps_coords)
            dfs = dfs.merge(df_gps, on="file_name", how="left")

            # converting gps coords to decimal
            dfs[["Latitude", "Longitude", "Elevation"]] = dfs.gps.apply(
                lambda x: self.format_gps(x)
            ).apply(pd.Series)
            dfs[["px_Latitude", "px_Longitude"]] = dfs.apply(
                self.get_detections_gps, axis=1
            )

        try:
            dfs.drop(
                columns=[
                    "image_id",
                    "iscrowd",
                    "segmentation",
                    "bbox",
                ],
                inplace=True,
            )
        except Exception:
            logger.info(
                "Tried to drop columns: ['image_id','iscrowd','segmentation','bbox']."
            )
            traceback.print_exc()

        return dfs

    def format_prediction(self, pred: dict, img_height: int, img_width: int):
        """Formatting the prediction to work with Label studio

        Args:
            pred (dict): prediction in coco format
            img_height (int): image height
            img_width (int): image width

        Returns:
            dict: Label studio formated prediction
        """
        x, y, width, height = pred["bbox"]
        label = pred["category_name"]
        score = pred["score"]

        template = {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "rectanglelabels": [label],
                "x": x / img_width * 100,
                "y": y / img_height * 100,
                "width": width / img_width * 100,
                "height": height / img_height * 100,
            },
            "score": score,
        }

        return template
