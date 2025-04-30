import pandas as pd
from datalabeling.ml import Detector

if __name__ == "__main__":
    # import geopy
    # from geopy import distance
    # from datalabeling.common.annotation_utils import GPSUtils, ImageProcessor

    # img_path = r"D:\paul_data\DJI_20231002100957_0001.JPG"

    # detector = Detector(path_to_weights=None)

    # results = detector.predict(image=None,
    #                  image_path=img_path,
    #                  inference_service_url='http://127.0.0.1:4141/docs',
    #                  return_gps=True
    #                 )

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

    # import requests, base64

    # with open(img_path, "rb") as f:
    #     img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # resp = requests.post(
    #     "http://127.0.0.1:4141/predict",
    #     json={
    #         "image": img_b64,
    #         "sahi_prostprocess": "NMS",
    #         "override_tilesize": None,  # tilesize to use for
    #         "postprocess_match_threshold": 0.5,
    #         "nms_iou": None,
    #         "return_as_decimal": True,
    #     },
    # )

    # detections = resp.json()

    from dotenv import load_dotenv

    load_dotenv(r"D:\datalabeling\.env")

    from datalabeling.ml import Annotator
    import os
    from pathlib import Path
    import torch
    from tqdm import tqdm

    from label_studio_sdk.client import LabelStudio

    use_sliding_window = True

    handler = (
        Annotator(  # path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
            #                     is_yolo_obb=True,
            #                     tilesize=640,
            #                     overlapratio=0.1,
            #                     use_sliding_window=use_sliding_window,
            #                     confidence_threshold=0.5,
            #                     # device="NPU", # "cpu", "cuda"
            #                     tag_to_append=f"-sahi:{use_sliding_window}",
            dotenv_path="../.env"
        )
    )

    project_id = 3  # insert correct project_id by loooking at the url
    # top_n=10
    # handler.upload_predictions(project_id=project_id,top_n=top_n)

    instances_count, images_count = handler.get_project_stats(
        project_id=project_id, annotator_id=0
    )

    # LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    # API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

    # ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # project = ls.projects.get(3)

    # tasks = ls.tasks.list(project=project.id,)
