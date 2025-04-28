import pandas as pd

if __name__ == "__main__":
    # import geopy
    # from geopy import distance
    # from datalabeling.common.annotation_utils import GPSUtils, ImageProcessor

    img_path = r"D:\paul_data\DJI_20231002100957_0001.JPG"

    # gps_coord, _ = GPSUtils.get_gps_coord(img_path, return_as_decimal=True)

    # gsd = ImageProcessor.get_gsd(
    #     image_path=img_path, sensor_height=None, flight_height=180
    # )

    # exif = GPSUtils.get_exif(img_path)

    # H = exif["ExifImageHeight"]
    # W = exif["ExifImageWidth"]

    # lat_center, lon_center, alt = gps_coord

    # px_lat, px_long = ImageProcessor.generate_pixel_coordinates(
    #     x=0,
    #     y=H / 2,
    #     lat_center=lat_center,
    #     lon_center=lon_center,
    #     W=W,
    #     H=H,
    #     gsd=gsd * 1e-2,
    # )

    # p_center = geopy.Point(latitude=lat_center, longitude=lon_center)

    # p_top_left = geopy.Point(latitude=px_lat, longitude=px_long)

    # d = distance.geodesic(p_center, p_top_left, ellipsoid="WGS-84").meters

    import requests, base64

    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(
        "http://127.0.0.1:4141/predict",
        json={
            "image": img_b64,
            "sahi_prostprocess": "NMS",
            "override_tilesize": None,  # tilesize to use for
            "postprocess_match_threshold": 0.5,
            "nms_iou": None,
            "return_as_decimal": True,
        },
    )

    detections = resp.json()
