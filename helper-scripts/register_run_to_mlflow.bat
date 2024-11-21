call .\activate_label-backend_env.bat

@REM start .\launch_mlflow_server.bat


@REM --use-sliding-window adding this flag will enabled sahi inference


call python ..\tools\register_model.py --exp-name "wildAI-detection" --model "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/d0dbe2b4cbe143258121a734edd9dca8/artifacts/weights/best.pt"^
        --model-name "obb-detector"^
        --tilesize 2000 --imgsz 1280 --confidence-threshold 0.1 --overlap-ratio 0.1^
        --is-yolo-obb --use-sliding-window

call conda deactivate