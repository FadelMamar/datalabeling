call .\activate_label-backend_env.bat

@REM start .\launch_mlflow_server.bat


@REM --use-sliding-window adding this flag will enabled sahi inference


call python ..\tools\register_model.py --exp-name "wildAI-detection" --model "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/e935d0ce21c64cfc87b4b921dcd4a142/artifacts/weights/best.pt"^
        --model-name "obb-detector"^
        --tilesize 1280 --confidence-threshold 0.1 --overlap-ratio 0.1^
        --is-yolo-obb

call conda deactivate