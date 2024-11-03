call .\activate_label-backend_env.bat

start .\launch_mlflow_server.bat

call python tools\register_model.py --model-name "detector" --name "wildAI-detection"^
             --model "../runs/mlflow/140168774036374062/0f0eb67413054fac9887cc5f6437e692/artifacts/weights/best.pt"

call conda deactivate