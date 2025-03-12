import mlflow
import torch
from sahi.predict import get_sliced_prediction
from ..annotator import Detector


class DetectorWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self,tilesize:int=640,
                 confidence_threshold:float=0.1,
                 overlap_ratio:float=0.1,
                 imgsz:int=640,
                 use_sliding_window:bool=True,
                 nms_iou:bool=0.5,
                 is_yolo_obb:bool=False,
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
        self.imgsz = imgsz
        self.sahi_postprocess=sahi_postprocess
        self.use_sliding_window=use_sliding_window
        self.nms_iou=nms_iou
        self.is_yolo_obb=is_yolo_obb

    def load_context(self, context):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.detection_model = Yolov8DetectionModel(model_path=context.artifacts['path'],
        #                                             confidence_threshold=self.confidence_threshold, #context.artifacts['confidence_threshold'],
        #                                             device=device)
        self.detection_model = Detector(path_to_weights=context.artifacts['path'],
                                        confidence_threshold=self.confidence_threshold,
                                        overlap_ratio=self.overlapratio,
                                        imgsz=self.imgsz,
                                        tilesize=self.tilesize,
                                        device=device,
                                        use_sliding_window=self.use_sliding_window,
                                        is_yolo_obb=self.is_yolo_obb)
        

    def predict(self, context, img):
        return self.detection_model.predict(img,return_coco=True,postprocess_match_threshold=self.nms_iou)

    