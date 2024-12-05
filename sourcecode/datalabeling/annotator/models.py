from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
# from label_studio_ml.utils import (get_env, get_local_path)
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import ultralytics
import torch
import json
from pathlib import Path
import logging
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import torch
import traceback

logger = logging.getLogger(__name__)


# https://github.com/obss/sahi/blob/main/sahi/models/yolov8.py
class Yolov8ObbDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        from ultralytics import YOLO

        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8-obb model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8-obb model.
        Args:
            model: Any
                A YOLOv8-obb model
        """

        self.model = model
        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        If predictions have masks, each prediction is a tuple like (boxes, masks).
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.

        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device}

        if self.image_size is not None:
            kwargs = {"imgsz": self.image_size, **kwargs}

        prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLOv8 expects numpy arrays to have BGR

        if self.has_mask:
            raise NotImplementedError("Not designed to handle masks.")

        else:  # If model doesn't do segmentation then no need to check masks
            prediction_result = [torch.cat([result.obb.xyxy,
                                            result.obb.conf.reshape(-1,1),
                                            result.obb.cls.reshape(-1,1)],dim=1) for result in prediction_result]

        self._original_predictions = prediction_result
        self._original_shape = image.shape

    @property
    def category_names(self):
        return self.model.names.values()

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return self.model.overrides["task"] == "segment"

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []
            if self.has_mask:
                raise NotImplementedError("Not designed to handle masks.")
            else:  # Only bounding boxes
                # process predictions
                for prediction in image_predictions.cpu().detach().numpy():
                    x1 = prediction[0]
                    y1 = prediction[1]
                    x2 = prediction[2]
                    y2 = prediction[3]
                    bbox = [x1, y1, x2, y2]
                    score = prediction[4]
                    category_id = int(prediction[5])
                    category_name = self.category_mapping[str(category_id)]

                    # fix negative box coords
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = max(0, bbox[2])
                    bbox[3] = max(0, bbox[3])

                    # fix out of image box coords
                    if full_shape is not None:
                        bbox[0] = min(full_shape[1], bbox[0])
                        bbox[1] = min(full_shape[0], bbox[1])
                        bbox[2] = min(full_shape[1], bbox[2])
                        bbox[3] = min(full_shape[0], bbox[3])

                    # ignore invalid predictions
                    if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                        logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                        continue

                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        # segmentation=None,
                        category_name=category_name,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
                object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

class Detector(object):

    def __init__(self,
                path_to_weights:str,
                confidence_threshold:float=0.1,
                overlap_ratio:float=0.1,
                tilesize:int|None=1280,
                imgsz:int=1280,
                device:str=None,
                use_sliding_window:bool=True,
                is_yolo_obb:bool=False):
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
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tilesize=tilesize
        self.imgsz=imgsz
        self.overlapratio=overlap_ratio
        self.use_sliding_window = use_sliding_window
        self.is_yolo_obb = is_yolo_obb
        logger.info( f"Computing device: {device}")
        if is_yolo_obb:
            self.detection_model = Yolov8ObbDetectionModel(model=YOLO(path_to_weights,task='obb'),
                                                            confidence_threshold=confidence_threshold,
                                                            image_size=self.imgsz,
                                                            device=device)
        else:
            self.detection_model = Yolov8DetectionModel(model=YOLO(path_to_weights,task='detect'),
                                                        confidence_threshold=confidence_threshold,
                                                        image_size=self.imgsz,
                                                        device=device,
                                                        )

    def predict(self, image:Image,
                return_coco:bool=True,
                sahi_prostprocess:float='NMS',
                override_tilesize:int=None,
                postprocess_match_threshold:float=0.5,
                nms_iou:float=None
                ):
        """Run sliced predictions

        Args:
            image (Image): input image

        Returns:
            dict: predictions in coco format
        """
        # if data is on AWS
        # r = urlparse(url, allow_fragments=False)
        # bucket_name = r.netloc
        # filename = r.path.lstrip('/')
        # with open('./tmp/s3_img.jpg','wb+') as f:
        #     S3.download_fileobj(bucket_name, filename, f)
        #     image = Image.open(f)

        if self.use_sliding_window:
            tilesize = override_tilesize if (self.tilesize is None) else self.tilesize
            result = get_sliced_prediction(image,
                                            self.detection_model,
                                            slice_height=tilesize,
                                            slice_width=tilesize,
                                            overlap_height_ratio=self.overlapratio,
                                            overlap_width_ratio=self.overlapratio,
                                            postprocess_type=sahi_prostprocess,
                                            postprocess_match_metric="IOU",
                                            verbose=1,
                                            postprocess_match_threshold=postprocess_match_threshold or nms_iou,
                                            )
        else:
            result = get_prediction(
                    image=image,
                    detection_model=self.detection_model,
                    shift_amount=[0, 0],
                    full_shape=None,
                    postprocess=None,
                    verbose=1,
                )
            # return self.yolo_results_to_coco(results=result[0],is_yolo_obb=self.is_yolo_obb)
    
        if return_coco:
            return result.to_coco_annotations()
        
        return result


    def yolo_results_to_coco(self,results:ultralytics.engine.results.Results,
                             is_yolo_obb:bool=True):

        if not is_yolo_obb:
            raise NotImplementedError("Supports only yolo-obb")
        
        coordinates = results.obb.xyxy.cpu().numpy().tolist()
        coco_results = list()

        for (x_min,y_min,x_max,y_max) in coordinates:
            
            template = {'image_id': None,
                    'bbox': [x_min, y_min, x_max-x_min, y_max-y_min], 
                    'score': results.obb.conf[0].item(),
                    'category_id': 0, 
                    'category_name': 'wildlife', 
                    'segmentation': [], 
                    'iscrowd': 0, 
                    'area': None}
            coco_results.append(template)
        
        return coco_results


    def predict_directory(self,path_to_dir:str,as_dataframe:bool=False,save_path:str=None):
        """Computes predictions on a directory

        Args:
            path_to_dir (str): path to directory with images
            as_dataframe (bool): returns results as pd.DataFrame
            save_path (str) : converts to dataframe and then save

        Returns:
            dict: a directory with the schema {image_path:prediction_coco_format}
        """
        results = {}
        for image_path in tqdm(Path(path_to_dir).iterdir()):
            pred = self.predict(Image.open(image_path),return_coco=True)
            results.update({str(image_path):pred})
        
        # returns as df or save
        if as_dataframe or (save_path is not None):
            results = self.get_pred_results_as_dataframe(results)

            if save_path is not None:
                try:
                    results.to_json(save_path,orient='records',indent=2)
                except Exception as e:
                    print('!!!Failed to save results as json!!!\n')
                    traceback.print_exc()

            return results
                        
        return results

    
    def get_pred_results_as_dataframe(self,results:dict[str:list]):

        df_results = pd.DataFrame.from_dict(results,orient='index')
        dfs = list()
        
        for i in tqdm(range(len(df_results)),desc='pred results as df'):
            df_i = pd.DataFrame.from_records(df_results.iloc[i,:].dropna().to_list())
            df_i['image_path'] = df_results.index[i]
            
            dfs.append(df_i)

        dfs = pd.concat(dfs,axis=0)
        dfs['x_min'] = dfs['bbox'].apply(lambda x: x[0])
        dfs['y_min'] = dfs['bbox'].apply(lambda x: x[1])
        dfs['bbox_w'] = dfs['bbox'].apply(lambda x: x[2])
        dfs['bbox_h'] = dfs['bbox'].apply(lambda x: x[3])
        dfs['x_max'] = dfs['x_min'] + dfs['bbox_w']
        dfs['y_max'] = dfs['y_min'] + dfs['bbox_h']

        try:
            dfs.drop(columns=['bbox','image_id','segmentation','iscrowd'],inplace=True)
        except Exception as e:
            print("Tried to drop columns: ['bbox','image_id','segmentation','iscrowd'].")
            traceback.print_exc()
            

        return dfs


    def format_prediction(self,pred:dict,img_height:int,img_width:int):
        """Formatting the prediction to work with Label studio

        Args:
            pred (dict): prediction in coco format
            img_height (int): image height
            img_width (int): image width

        Returns:
            dict: Label studio formated prediction
        """
        x, y, width, height = pred['bbox']
        label = pred['category_name']
        score = pred['score']

        template = {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    'value': {
                        'rectanglelabels': [label],
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': width / img_width * 100,
                        'height': height / img_height * 100
                    },
                    'score': score
        }

        return template

    def train(self, dataloader):
       raise NotImplementedError('Not implemented.')
       pass