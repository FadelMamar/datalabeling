from sahi.models.yolov8 import Yolov8DetectionModel
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
import torch
from PIL import Image
from time import time
from pathlib import Path
import mlflow
import json
from dotenv import load_dotenv
from label_studio_ml.utils import get_local_path
from label_studio_sdk import Client
from tqdm import tqdm
import os
import logging

class Annotator(object):

    def __init__(self,
                dotenv_path:str=None,
                path_to_weights:str=None,
                mlflow_model_alias:str="start",
                mlflow_model_name:str="detector",
                mlflow_model_version:str=None,
                confidence_threshold:float=0.1):
        """_summary_

        Args:
            path_to_weights (str, optional): path to weights. Defaults to None.
            mlflow_model_alias (str, optional): mflow registered model alias. Defaults to "start".
            mlflow_model_name (str, optional): model name. Defaults to "detector".
            mlflow_model_version (str, optional): registered model version. Defaults to None.
            confidence_threshold (float, optional): Detection threshold. Defaults to 0.35.
        """

        # Load environment variables
        if dotenv_path is not None:
            load_dotenv(dotenv_path=dotenv_path)
            # Connect to the Label Studio API and check the connection
            LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL')
            API_KEY = os.getenv("LABELSTUDIO-API-KEY")
            self.labelstudio_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
        else:
            logging.warning(msg="Pass argument `dotenv_path` to access label studio API")

        ## Load model from path
        self.tilesize=640
        self.overlapratio=0.1
        self.sahi_prostprocess='NMS'

        self.path_to_weights = path_to_weights
        if self.path_to_weights is None:
            TRACKING_URI="http://localhost:5000"
            mlflow.set_tracking_uri(TRACKING_URI)
            client = mlflow.MlflowClient()
            name = mlflow_model_name
            alias = mlflow_model_alias
            version = client.get_model_version_by_alias(name=name,alias=alias).version
            self.modelversion = f'{name}:{version}'
            self.modelURI = f'models:/{name}/{version}'
            self.model = mlflow.pyfunc.load_model(self.modelURI)
            # print('Device:',self.model.detection_model.device)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = Yolov8DetectionModel(
                                                        model=YOLO(path_to_weights,task='detect'),
                                                        confidence_threshold=confidence_threshold,
                                                        image_size=self.tilesize,
                                                        device=device,
                                                        )
            self.modelversion = Path(path_to_weights).stem
            # print('Device:', device)
        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"
        if mlflow_model_version is not None:
            self.modelversion = mlflow_model_version

    def predict(self, image:bytearray) -> dict:
        """_summary_

        Args:
            image (bytearray): object of PIL.Image.open

        Returns:
            dict: prediction in coco annotation format
        """
        if self.path_to_weights is not None:
            result = get_sliced_prediction(image,
                                            self.model,
                                            slice_height=self.tilesize,
                                            slice_width=self.tilesize,
                                            overlap_height_ratio=self.overlapratio,
                                            overlap_width_ratio=self.overlapratio,
                                            postprocess_type=self.sahi_prostprocess,
                                            )
            return result.to_coco_annotations()


        return  self.model.predict(image)

    def format_prediction(self,pred:dict,img_height:int,img_width:int) -> dict:
        """_summary_

        Args:
            pred (dict): prediction in coco format
            img_height (int): image height
            img_width (int): image width

        Returns:
            dict: Label studio formated prediction
        """
        # formatting the prediction to work with Label studio
        x, y, width, height = pred['bbox']
        label = pred['category_name']
        score = pred['score']
        if not isinstance(score,float):
            score = 0.0
        template = {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": self.label_type,
                    "original_width":img_width,
                    "original_height":img_height,
                    "image_rotation":0,
                    'value': {
                        self.label_type: [label,],
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': width / img_width * 100,
                        'height': height / img_height * 100,
                        'rotation':0
                    },
                    'score': score
        }
        return template

    def upload_predictions(self,project_id:int,top_n:int=None)->None:
        """Uploads predictions using label studio API.
        Make sure to set the API key and url inside .env

        Args:
            project_id (int): project id from Label studio
            top_n (int): top n tasks to be uploaded in descending order of task_id.
        """
        # Select project
        project = self.labelstudio_client.get_project(id=project_id)

        # Upload predictions for each task
        tasks = project.get_tasks()
        if top_n > 0:
            tasks = sorted(tasks,key=lambda x:x['id'])[:top_n]
        for task in tqdm(tasks,desc="Uploading predictions"):
            task_id = task['id']
            img_url = task['data']['image']
            img_path = get_local_path(img_url)
            img = Image.open(img_path)
            prediction = self.predict(img)
            img_width, img_height = img.size
            formatted_pred = [self.format_prediction(pred,
                                                        img_height=img_height,
                                                        img_width=img_width) for pred in prediction]
            conf_scores = [pred['score'] for pred in prediction]
            max_score = 0.0
            if len(conf_scores)>0:
                max_score = max(conf_scores)
            project.create_prediction(task_id=task_id,
                                score=max_score,
                                result=formatted_pred,
                                model_version=self.modelversion)

    def build_upload_json(self,path_img_dir:str,
                          root:str,
                        #   project_id:int=None,
                          pattern="*.JPG",
                          bulk_predictions:list[dict]=None,
                          save_json_path:str=None) -> list[dict]:
        """_summary_

        Args:
            path_img_dir (str): directory with images of interest
            root (str): root specified in environment variable LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT or LOCAL_FILES_DOCUMENT_ROOT.
            pattern (str, optional): image extension pattern. It is passed to glob. Defaults to "*.JPG".
            bulk_predictions (dict, optional): dictionary with pre-computed detections. The key are the image file names. Defaults to None.
            save_json_path (str, optional): path to which the json file should be saved. Defaults to None.

        Returns:
            list[dict]: _description_
        """
        directory_preds = list()
        # Select project
        # project = self.labelstudio_client.get_project(id=project_id)
        # tasks = project.get_tasks()

        # Upload predictions for each task
        # for task in tasks:
        for image_path in Path(path_img_dir).glob(pattern):
            img_path_as_posix = image_path.relative_to(Path(root)).as_posix()
            # task_id = task['id']
            # img_url = task['data']['image']
            img_url = f"/data/local-files/?d={img_path_as_posix}"
            pred = {
                        "data": {"image" : img_url},
                        "predictions":[],
                    }
            # pred['id'] = task_id
            image_path = Path(get_local_path(img_url))
            # get predictions
            if bulk_predictions is None:
                start = time()
                image = Image.open(image_path)
                predictions = self.predict(image)
                print(f'Prediction time:{time() - start:.3f} seconds.')
                # format predictions
                img_width, img_height = image.size
                formatted_pred = [self.format_prediction(pred,
                                                        img_height=img_height,
                                                        img_width=img_width) for pred in predictions]
            else:
                predictions = bulk_predictions[image_path.name]
                formatted_pred = [self.format_prediction(pred,
                                                        img_height=pred['height'],
                                                        img_width=pred['width']) for pred in predictions]
            conf_scores = [pred['score'] for pred in predictions]
            # store predictions
            if len(conf_scores)>0:
                pred['predictions'].append({'result':formatted_pred,
                                            'model_version':self.modelversion,
                                            'score':max(conf_scores),
                                            }
                                            )
            else:
                pred['predictions'].append({'result':formatted_pred,
                                            'model_version':self.modelversion,
                                            'score':0.0
                                            }
                                            )
            # update buffer
            directory_preds.append(pred)

        if save_json_path is not None:
            with open(Path(save_json_path),'w') as file:
                json.dump(directory_preds,file,indent=2)

        return directory_preds