from label_studio_ml.model import LabelStudioMLBase
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
from label_studio_ml.utils import (get_env, get_local_path)
from PIL import Image
from ultralytics import YOLO
import torch


class Detector(object):

    def __init__(self,
                path_to_weights:str,
                confidence_threshold:float=0.1):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tilesize=640
        self.overlapratio=0.1
        self.sahi_prostprocess='NMS'
        print('Device:', device)
        self.detection_model = Yolov8DetectionModel(
                                                    # model_path=path_to_weights,
                                                    model=YOLO(path_to_weights,task='detect'),
                                                    confidence_threshold=confidence_threshold,
                                                    image_size=self.tilesize,
                                                    device=device,
                                                    )

    def predict(self, image:Image):
        """Run sliced predictions

        Args:
            image (Image): _description_

        Returns:
            _type_: _description_
        """
        # if data is on AWS
        # r = urlparse(url, allow_fragments=False)
        # bucket_name = r.netloc
        # filename = r.path.lstrip('/')
        # with open('./tmp/s3_img.jpg','wb+') as f:
        #     S3.download_fileobj(bucket_name, filename, f)
        #     image = Image.open(f)

        result = get_sliced_prediction(image,
                                        self.detection_model,
                                        slice_height=self.tilesize,
                                        slice_width=self.tilesize,
                                        overlap_height_ratio=self.overlapratio,
                                        overlap_width_ratio=self.overlapratio,
                                        postprocess_type=self.sahi_prostprocess,
                                        )

        return result.to_coco_annotations()

    def format_prediction(self,pred:dict,img_height:int,img_width:int):
        """Formatting the prediction to work with Label studio

        Args:
            pred (Dict): _description_
            img_height (int): _description_
            img_width (int): _description_

        Returns:
            _type_: _description_
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