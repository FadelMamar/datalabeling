call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

@REM :: add 

@REM :: Example builds detection dataset
call uv run  tools\build_dataset.py --obb-to-yolo --skip --create-yolo-seg-dir --copy-images --data-config-yaml "data\dataset_identification-detection.yaml"^
    --sam-model-path "base_models_weights\sam2.1_l.pt"
