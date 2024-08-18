from sahi.models.yolov8 import Yolov8DetectionModel
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
import torch
from PIL import Image
from time import time
from pathlib import Path
import mlflow
import json

class Annotator(object):

    def __init__(self,
                path_to_weights:str=None,
                mlflow_model_alias:str="start",
                mlflow_model_name:str="detector",
                mlflow_model_version:str=None,
                confidence_threshold:float=0.35):
        
        ## Load model from path
        self.tilesize=640
        self.overlapratio=0.1
        self.sahi_prostprocess='NMS'
        
        self.path_to_weights = path_to_weights
        if self.path_to_weights is None:
            ## Load  from mlflow
            TRACKING_URI="http://localhost:5000"
            mlflow.set_tracking_uri(TRACKING_URI)
            client = mlflow.MlflowClient()
            name = mlflow_model_name
            alias = mlflow_model_alias
            version = client.get_model_version_by_alias(name=name,alias=alias).version
            self.modelversion = f'{name}:{version}'
            self.modelURI = f'models:/{name}/{version}'
            self.model = mlflow.pyfunc.load_model(self.modelURI)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print('Device:', device)
            self.model = Yolov8DetectionModel(
                                                        model=YOLO(path_to_weights,task='detect'),
                                                        confidence_threshold=confidence_threshold,
                                                        image_size=self.tilesize,
                                                        device=device,
                                                        )
            self.modelversion = Path(path_to_weights).stem
            
        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"
        if mlflow_model_version is not None:
            self.modelversion = mlflow_model_version

    def predict(self, image:bytearray):

        if self.path_to_weights is None:
            return self.model.predict(image)
        
        result = get_sliced_prediction(image,
                                        self.model,
                                        slice_height=self.tilesize,
                                        slice_width=self.tilesize,
                                        overlap_height_ratio=self.overlapratio,
                                        overlap_width_ratio=self.overlapratio,
                                        postprocess_type=self.sahi_prostprocess,
                                        )
        return result.to_coco_annotations()

    def format_prediction(self,pred:dict,img_height:int,img_width:int):
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

    def build_upload_json(self,path_img_dir:str,root:str,
                          pattern="*.JPG",
                          bulk_predictions:list[dict]=None,
                          save_json_path:str=None):

        directory_preds = list()

        for image_path in Path(path_img_dir).glob(pattern):
            d=image_path.relative_to(Path(root)).as_posix()
            pred = { 
                        "data": {"image" : f"/data/local-files/?d={d}"},
                        "predictions":[],
                    }
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