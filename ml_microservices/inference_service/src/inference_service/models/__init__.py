import logging
import traceback
import geopy
import pandas as pd
import torch
from PIL import Image
from sahi.predict import get_prediction, get_sliced_prediction
import mlflow
from tqdm import tqdm

from ..core.utils import GPSUtils, ImageProcessor

logger = logging.getLogger(__name__)


class Detector(object):
    def __init__(
        self,
        mlflow_model_name: str,
        mlflow_model_alias: str,
        use_sliding_window: bool = True,
        confidence_threshold: float = 0.15,
        overlap_ratio: float = 0.2,
        tilesize: int | None = 960,
        imgsz: int = 960,
        device: str = None,
        tracking_url: str = "http://mlflow_service:5000",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detection_model = None

        self.sahi_prostprocess = "NMS"

        self.confidence_threshold = confidence_threshold
        self.overlap_ratio = overlap_ratio
        self.tilesize = tilesize
        self.imgsz = imgsz
        self.use_sliding_window = use_sliding_window

        self.tracking_url = tracking_url
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_alias = mlflow_model_alias

        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

        self.model = None
        self.modelURI = None
        self.modelversion = None

    def _set_model(self):
        mlflow.set_tracking_uri(self.tracking_url)
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(
            name=self.mlflow_model_name, alias=self.mlflow_model_alias
        ).version
        self.modelversion = f"{self.mlflow_model_name}:{version}"
        self.modelURI = f"models:/{self.mlflow_model_name}/{version}"
        self.model = mlflow.pyfunc.load_model(self.modelURI)

    def predict(
        self,
        image: Image.Image = None,
        image_path: str = None,
        return_gps: bool = False,
        return_coco: bool = False,
        sahi_prostprocess: float = "NMS",
        override_tilesize: int = None,
        postprocess_match_threshold: float = 0.5,
        nms_iou: float = None,
        verbose: int = 0,
    ):
        """Run sliced predictions

        Args:
            image (Image): input image

        Returns:
            dict: predictions in coco format
        """

        if self.model is None:
            self._set_model()

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
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
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

            lat_center, lon_center, alt = x.img_Latitude, x.img_Longitude, x.Elevation

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
        for file_name, value in results.items():
            for v in value:
                if "gps_coords" not in v.keys():
                    unravel_dict.append(
                        {"file_name": file_name, "value": v}
                    )  # = v #,results['gps_coords']
                elif return_gps:
                    gps_coords.append({"gps": v["gps_coords"], "file_name": file_name})

        df_results = pd.DataFrame.from_dict(unravel_dict)

        dfs = list()
        for i in tqdm(range(len(df_results)), desc="pred results as df"):
            df_i = pd.DataFrame.from_records(df_results.iloc[i, 1:].to_list())
            df_i["file_name"] = df_results.iat[i, 0]
            dfs.append(df_i)

        dfs = pd.concat(dfs, axis=0)  # .dropna(thresh=5)
        # bbox is in coco format (x_min,y_min,w,h)
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
            dfs[["img_Latitude", "img_Longitude", "Elevation"]] = dfs.gps.apply(
                lambda x: self.format_gps(x)
            ).apply(pd.Series)
            dfs[["Latitude", "Longitude"]] = dfs.apply(self.get_detections_gps, axis=1)

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
