import geopy
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
import base64, requests
import logging, traceback
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import utm

logger = logging.getLogger(__name__)


class GPSUtils:
    @staticmethod
    def get_exif(file_name: str, image: Image = None) -> dict | None:
        if image is None:
            with Image.open(file_name) as img:
                exif_data = img._getexif()
        else:
            exif_data = image._getexif()

        if exif_data is None:
            return None

        extracted_exif = dict()
        for k, v in exif_data.items():
            extracted_exif[TAGS.get(k)] = v

        return extracted_exif

    @staticmethod
    def get_gps_info(labeled_exif: dict) -> dict | None:
        # https://exiftool.org/TagNames/GPS.html

        gps_info = labeled_exif.get("GPSInfo", None)

        if gps_info is None:
            return None

        info = {GPSTAGS.get(key, key): value for key, value in gps_info.items()}

        info["GPSAltitude"] = info["GPSAltitude"].__repr__()

        # convert bytes types
        for k, v in info.items():
            if isinstance(v, bytes):
                info[k] = list(v)

        return info

    @staticmethod
    def get_gps_coord(
        file_name: str,
        image: Image = None,
        altitude: str = None,
        return_as_decimal: bool = False,
    ) -> tuple | None:
        extracted_exif = GPSUtils.get_exif(file_name=file_name, image=image)

        if extracted_exif is None:
            return None

        gps_info = GPSUtils.get_gps_info(extracted_exif)

        if gps_info is None:
            return None

        if gps_info.get("GPSAltitudeRef", None):
            altitude_map = {
                0: "Above Sea Level",
                1: "Below Sea Level",
                2: "Positive Sea Level (sea-level ref)",
                3: "Negative Sea Level (sea-level ref)",
            }

            # map GPSAltitudeRef
            try:
                gps_info["GPSAltitudeRef"] = altitude_map[gps_info["GPSAltitudeRef"]]
            except:
                gps_info["GPSAltitudeRef"] = altitude_map[gps_info["GPSAltitudeRef"][0]]

        # rewite latitude
        gps_coords = dict()
        for coord in ["GPSLatitude", "GPSLongitude"]:
            degrees, minutes, seconds = gps_info[coord]
            ref = gps_info[coord + "Ref"]
            gps_coords[coord] = f"{degrees} {minutes}m {seconds}s {ref}"

        coords = gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"]

        if altitude is None:
            alt = f"{gps_info['GPSAltitude']}m"
        else:
            alt = altitude

        coords = (
            gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"] + " " + alt
        )
        if return_as_decimal:
            lat, long, alt = geopy.Point.from_string(coords)
            coords = lat, long, alt * 1e3

        return coords, gps_info


class ImageProcessor:
    @staticmethod
    def get_gsd(
        image_path: str,
        image: Image.Image | None = None,
        sensor_height: float = None,
        flight_height: int = 180,
    ):
        ##-- Sensor heights
        sensor_heights = dict(ZenmuseP1=24)

        ##-- Extract exif
        exif = GPSUtils.get_exif(file_name=image_path, image=image)

        if sensor_height is None:
            sensor_height = sensor_heights[exif["Model"]]

        ##-- Compute gsd
        focal_length = exif["FocalLength"] * 0.1  # in cm
        image_height = exif["ExifImageHeight"]  # in px
        sensor_height = sensor_height * 0.1  # in cm
        flight_height = flight_height * 1e2  # in cm

        gsd = flight_height * sensor_height / (focal_length * image_height)

        return round(gsd, 3)

    @staticmethod
    def generate_pixel_coordinates(x, y, lat_center, lon_center, W, H, gsd=0.026):
        # Convert center to UTM
        easting_center, northing_center, zone_num, zone_let = utm.from_latlon(
            lat_center, lon_center
        )

        # Calculate offsets
        delta_x = (x - W / 2) * gsd
        delta_y = (H / 2 - y) * gsd  # Invert y-axis

        # Compute UTM
        easting = easting_center + delta_x
        northing = northing_center + delta_y

        # Convert back to lat/lon
        lat, lon = utm.to_latlon(easting, northing, zone_num, zone_let)

        return lat, lon


class Detector(object):
    def __init__(
        self,
        alias: str = None,
        inference_service_url: str = "http://127.0.0.1:4141/predict",
        return_gps: bool = True,
        timeout=3 * 60,
        postprocess_match_threshold=0.5,
    ):
        self.alias = alias
        self.inference_service_url = inference_service_url
        self.return_gps = return_gps
        self.timeout = timeout
        self.postprocess_match_threshold = postprocess_match_threshold

    def _predict_url(
        self,
        image_path: str,
    ):
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "image": img_b64,
            "sahi_prostprocess": "NMS",
            "override_tilesize": None,  # tilesize to use for
            "postprocess_match_threshold": self.postprocess_match_threshold,
            "nms_iou": None,
            "return_as_decimal": False,
        }

        resp = requests.post(
            self.inference_service_url,
            json=payload,
            timeout=self.timeout,
        ).json()

        detections = resp["detections"]
        image_gps = resp["image_gps"]

        if self.return_gps:
            return detections, image_gps

        return detections

    def predict_directory(
        self,
        path_to_dir: str = None,
        images_paths: list[str] = None,
        return_gps: bool = True,
        as_dataframe: bool = True,
    ) -> dict | pd.DataFrame:
        assert (path_to_dir is None) + (images_paths is None) < 2, (
            "Both should not be given."
        )
        results = {}
        paths = images_paths or list(Path(path_to_dir).iterdir())
        for image_path in tqdm(paths, desc="Computing predictions..."):
            try:
                pred = self._predict_url(
                    image_path=image_path,
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
        if as_dataframe:
            results = self.get_pred_results_as_dataframe(results, return_gps=return_gps)

        return results

    def _format_gps(self, gps_coord: str):
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
                lambda x: self._format_gps(x)
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
