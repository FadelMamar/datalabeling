import mlflow
import torch
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
from ..annotator import Yolov8ObbDetectionModel


class DetectorWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self,tilesize:int=640,
                 confidence_threshold:float=0.1,
                 overlap_ratio:float=0.1,
                 sahi_postprocess:str='NMS'):
        """_summary_

        Args:
            tilesize (int, optional): _description_. Defaults to 640.
            confidence_threshold (float, optional): _description_. Defaults to 0.1.
            overlap_ratio (float, optional): _description_. Defaults to 0.1.
            sahi_postprocess (str, optional): _description_. Defaults to 'NMS'.
        """
        super(DetectorWrapper,self).__init__()
        self.tilesize=tilesize
        self.confidence_threshold=confidence_threshold
        self.overlapratio=overlap_ratio
        self.sahi_postprocess=sahi_postprocess

    def load_context(self, context):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detection_model = Yolov8DetectionModel(model_path=context.artifacts['path'],
                                                    confidence_threshold=self.confidence_threshold, #context.artifacts['confidence_threshold'],
                                                    device=device)
        

    def predict(self, context, img):
        result = get_sliced_prediction(img, 
                                        self.detection_model,
                                        slice_height=self.tilesize,
                                        slice_width=self.tilesize,
                                        overlap_height_ratio=self.overlapratio,
                                        overlap_width_ratio=self.overlapratio,
                                        postprocess_type=self.sahi_postprocess,
                                        ) 

        return result.to_coco_annotations()

class ObbDetectorWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self,tilesize:int=640,
                 confidence_threshold:float=0.1,
                 overlap_ratio:float=0.1,
                 sahi_postprocess:str='NMS'):
        """_summary_

        Args:
            tilesize (int, optional): _description_. Defaults to 640.
            confidence_threshold (float, optional): _description_. Defaults to 0.1.
            overlap_ratio (float, optional): _description_. Defaults to 0.1.
            sahi_postprocess (str, optional): _description_. Defaults to 'NMS'.
        """
        super(ObbDetectorWrapper,self).__init__()
        self.tilesize=tilesize
        self.confidence_threshold=confidence_threshold
        self.overlapratio=overlap_ratio
        self.sahi_postprocess=sahi_postprocess

    def load_context(self, context):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detection_model = Yolov8ObbDetectionModel(model_path=context.artifacts['path'],
                                                    confidence_threshold=self.confidence_threshold, #context.artifacts['confidence_threshold'],
                                                    device=device)
    
    def predict(self, context, img):
        result = get_sliced_prediction(img, 
                                        self.detection_model,
                                        slice_height=self.tilesize,
                                        slice_width=self.tilesize,
                                        overlap_height_ratio=self.overlapratio,
                                        overlap_width_ratio=self.overlapratio,
                                        postprocess_type=self.sahi_postprocess,
                                        ) 

        return result.to_coco_annotations()
    