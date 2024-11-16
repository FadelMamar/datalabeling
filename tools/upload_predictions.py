# load env variables!
from dotenv import load_dotenv
DOTENV_PATH='../.env'
print("loading env variables:", load_dotenv(DOTENV_PATH))

import fire


def main(project_id:int,
         path_to_weights: str,
         is_yolo_obb:bool,
         tilesize:int, 
         overlapratio:float,
         confidence_threshold:float,
         device:str):
    

    from datalabeling.annotator import Annotator

    # using path_to_weights
    # go to ultralytics.nn.autobackend to modify ov_compiled device to "AUTO:NPU,GPU,CPU" + enable caching of model
    handler = Annotator(path_to_weights=path_to_weights,
                        is_yolo_obb=is_yolo_obb,
                        tilesize=tilesize,
                        overlapratio=overlapratio,
                        confidence_threshold=confidence_threshold,
                        device=device,
                        dotenv_path=DOTENV_PATH)
    
    handler.upload_predictions(project_id=project_id,top_n=0)


if __name__ == "__main__":
    fire.Fire(main)

