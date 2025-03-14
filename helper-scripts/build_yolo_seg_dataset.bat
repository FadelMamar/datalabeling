call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools"

@REM :: add 

@REM :: Example builds detection dataset
call uv run  build_dataset.py --obb-to-yolo --skip --create-yolo-seg-dir --copy-images --data-config-yaml ""^
    --sam-model-path "./sam2.1_l.pt"