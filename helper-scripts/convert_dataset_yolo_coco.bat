call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: yolo to obb
call python  tools\build_dataset.py --obb-to-yolo --yolo-to-coco --data-config-yaml "data\dataset_identification-detection.yaml" ^
     --coco-output-dir "D:\PhD\Data per camp\DetectionDataset\Identification-split\coco-dataset" --skip

call python  tools\build_dataset.py --obb-to-yolo --yolo-to-coco --data-config-yaml "data\dataset_identification.yaml" ^
    --coco-output-dir "D:\PhD\Data per camp\IdentificationDataset\coco-dataset" --skip

@REM call deactivate
