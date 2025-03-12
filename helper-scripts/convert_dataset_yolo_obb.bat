call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: yolo to obb
@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_conversion.yaml" --skip

:: obb to yolo
@REM call python  tools\build_dataset.py --obb-to-yolo --data-config-yaml "data\dataset_conversion.yaml" --skip

call deactivate
