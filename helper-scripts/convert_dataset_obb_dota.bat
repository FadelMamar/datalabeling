call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

:: yolo to obb
@REM call uv run  tools\build_dataset.py --yolo-to-obb --obb-to-dota --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --skip --clear-dota-labels

@REM call uv run  tools\build_dataset.py --yolo-to-obb --obb-to-dota --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --skip --clear-dota-labels

@REM call deactivate
