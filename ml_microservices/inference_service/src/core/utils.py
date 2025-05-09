import geopy
import utm
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
# from sahi.slicing import slice_coco
# from skimage.io import imread, imsave
# from tqdm import tqdm


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
