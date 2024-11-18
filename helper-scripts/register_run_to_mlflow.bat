call .\activate_label-backend_env.bat

start .\launch_mlflow_server.bat

call python tools\register_model.py --model-name "obb-detector" --is-yolo-obb True^
         --name "wildAI-detection"^
         --model "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/e935d0ce21c64cfc87b4b921dcd4a142/artifacts/weights/best.pt"

call conda deactivate