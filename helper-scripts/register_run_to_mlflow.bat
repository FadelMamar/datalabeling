call .\activate_label-backend_env.bat

@REM start .\launch_mlflow_server.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

@REM --use-sliding-window adding this flag will enabled sahi inference


call python tools\register_model.py --exp-name "wildAI-detection" --model "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/57daf3bcd99b4dd4b040cb4f8670960c/artifacts/weights/best.pt"^
        --model-name "obb-detector"^
        --tilesize 1280 --imgsz 1280 --confidence-threshold 0.1 --overlap-ratio 0.15^
        --is-yolo-obb --use-sliding-window

call conda deactivate