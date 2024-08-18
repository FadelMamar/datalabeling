from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
# from sahi.models.yolov8 import Yolov8DetectionModel
# from sahi.predict import get_sliced_prediction
from label_studio_ml.utils import get_local_path
from PIL import Image
# from ultralytics import YOLO
# import boto3
# import torch
# from pathlib import Path
# from urllib.parse import urlparse
import mlflow
# from requests.auth import HTTPBasicAuth
# import hashlib
# import os
import time
from datalabeling.annotator import Detector

# Authenticate AWS
# PROFILE_NAME = 'my-profile' #TODO: update with your profile
# MY_SESSION = boto3.session.Session(profile_name=PROFILE_NAME)
# S3 = MY_SESSION.client('s3')

#Mlflow
TRACKING_URI="http://localhost:5000"


class NewModel(LabelStudioMLBase):

    def __init__(self,**kwargs):
        super(NewModel, self).__init__(**kwargs)

        # pre-initialitzation of variables
        # _, schema = list(self.parsed_label_config.items())[0]
        self.from_name = "label"
        self.to_name = "image"
        self.value = "image"
        # self.labels = schema['labels']

        # model from mlflow registry
        mlflow.set_tracking_uri(TRACKING_URI)
        client = mlflow.MlflowClient()
        name = 'detector'
        alias = 'pt'
        version = client.get_model_version_by_alias(name=name,alias=alias).version
        self.modelversion = f'{name}:{version}'
        self.modelURI = f'models:/{name}/{version}'
        self.model = 0
        
        # Load localizer change model path to match
        self.model = Detector(path_to_weights="/Users/sfadel/Desktop/datalabeling/best.pt",
                            confidence_threshold=0.1)

    def __format_prediction(self,pred:Dict,img_height:int,img_width:int):
        # formatting the prediction to work with Label studio
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

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        # if self.model is None:
        #     self.model = mlflow.pyfunc.load_model(self.modelURI)
        # print(f'''\
        # * Run prediction on {tasks}
        # * Received context: {context}
        # * Project ID: {self.project_id}
        # * Label config: {self.label_config}
        # * Parsed JSON Label config: {self.parsed_label_config}''')
        if self.model == 0:
            self.model = mlflow.pyfunc.load_model(self.modelURI)

        # get predictions for every task
        preds = list()
        
        self.model = mlflow.pyfunc.load_model(self.modelURI)

        for task in tasks:
            img_url = task['data'][self.value]
            image = Image.open(get_local_path(img_url))
            start = time.time()
            predictions = self.model.predict(image)
            print(f'Prediction time:{time.time() - start:.3f} seconds.')
            img_width, img_height = image.size
            formatted_pred = [self.__format_prediction(pred,
                                                       img_height=img_height,
                                                       img_width=img_width) for pred in predictions]
            conf_scores = [pred['score'] for pred in predictions]
            if len(conf_scores)>0:
                preds.append({'result':formatted_pred,
                            'model_version':self.modelversion,
                            'task':task['id'],
                            'score':min(conf_scores),
                            }
                            )
            else:
                preds.append({'result':formatted_pred,
                            'model_version':self.modelversion,
                            'task':task['id'],
                            }
                            )


        return preds

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        # old_data = self.get('my_data')
        # old_model_version = self.get('model_version')
        # print(f'Old data: {old_data}')
        # print(f'Old model version: {old_model_version}')

        # store new data to the cache
        # self.set('my_data', 'my_new_data_value')
        # self.set('model_version', 'my_new_model_version')
        # print(f'New data: {self.get("my_data")}')
        # print(f'New model version: {self.get("model_version")}')

        # print('fit() completed successfully.')

        pass

