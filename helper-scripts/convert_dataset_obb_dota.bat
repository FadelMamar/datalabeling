call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: yolo to obb
call python  tools\build_dataset.py --yolo-to-obb --obb-to-dota --data-config-yaml "data\data_config.yaml" --skip --clear-dota-labels

@REM call deactivate
