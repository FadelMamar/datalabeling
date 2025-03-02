
from ultralytics import YOLO
# from pathlib import Path
from datalabeling.train import remove_label_cache



if __name__ == "__main__":
# Load yolov8s-obb
    paths = [
            "../runs/mlflow/140168774036374062/b883bd2b31f94f29807ea3b94e8ff8fc/artifacts/weights/best.pt", # Identification
            "../runs/mlflow/140168774036374062/4291304920cd40c28ff8456684045983/artifacts/weights/best.pt"  # Detection
            ]


    dataconfigs = [
                    r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml",
                r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification-detection.yaml"
                ]

    imgsz = 800
    iou_threshold=0.45
    conf_threshold=0.235
    splits = [
              "val", 
              "test",
            ]

    # remove label.cache files
    for dataconfig in dataconfigs:
        remove_label_cache(data_config_yaml=dataconfig)

    for split in splits:
        for path,dataconfig in zip(paths,dataconfigs):
            print("\n",'-'*20,split,'-'*20)
            model = YOLO(path)
            model.info()
            
            # Customize validation settings
            validation_results = model.val(data=dataconfig,
                                            imgsz=imgsz,
                                            batch=64,
                                            split=split,
                                            conf=conf_threshold,
                                            iou=iou_threshold,
                                            device="cuda"
                                        )

