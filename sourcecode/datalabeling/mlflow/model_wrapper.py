import mlflow
import torch
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction


class DetectorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detection_model = Yolov8DetectionModel(model_path=context.artifacts['path'],
                                                    confidence_threshold=0.4, #context.artifacts['confidence_threshold'],
                                                    device=device)
        self.tilesize=640
        self.overlapratio=0.1
        self.sahi_prostprocess='NMS'

    def predict(self, context, img):
        result = get_sliced_prediction(img, 
                                        self.detection_model,
                                        slice_height=self.tilesize,
                                        slice_width=self.tilesize,
                                        overlap_height_ratio=self.overlapratio,
                                        overlap_width_ratio=self.overlapratio,
                                        postprocess_type=self.sahi_prostprocess,
                                        ) 

        return result.to_coco_annotations()