call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

:: yolo to obb
@REM call python  tools\build_dataset.py --obb-to-yolo --yolo-to-coco --clear-coco-dir --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM      --coco-output-dir "D:\PhD\Data per camp\DetectionDataset\Identification-split\coco-dataset" --skip

@REM call python  tools\build_dataset.py --obb-to-yolo --yolo-to-coco --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --coco-output-dir "D:\PhD\Data per camp\IdentificationDataset\coco-dataset" --skip --clear-coco-dir

@REM call deactivate
