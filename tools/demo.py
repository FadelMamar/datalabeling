# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:36:09 2025

@author: Machine Learning
"""

from ultralytics import YOLO
from PIL import Image
import time
import numpy as np
from datalabeling.annotator import Detector, Annotator
from dotenv import load_dotenv
from pathlib import Path

load_dotenv('../.env')

# provide correct alias, "pt", "onnx"
aliases = ["version18",]
project_id = 97 # insert correct project_id by loooking at the url

# for alias in aliases:
name = "obb-detector" # detector, "obb-detector"
alias = aliases[0]

handler = Annotator(mlflow_model_alias=alias,
                    mlflow_model_name=name,
                    confidence_threshold=0.15,
                    is_yolo_obb=name.strip() == "obb-detector",
                    dotenv_path="../.env")


image_paths = list(Path(r"D:\PhD\Data per camp\Dry season\Leopard rock\Camp 1-8\Rep 1\DJI_202310021005_001").glob('*.jpg'))
len(image_paths)

results = handler.model.unwrap_python_model().detection_model.predict_directory(path_to_dir = None,
                                                                                images_paths = image_paths[:5],
                                                                                return_gps = True,
                                                                                return_coco = False,
                                                                                as_dataframe = True,
                                                                                save_path = None
                                                                            )