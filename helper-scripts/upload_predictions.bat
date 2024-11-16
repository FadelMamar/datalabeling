call .\activate_label-backend_env.bat

call cd ..\tools

call python upload_predictions.py 3 "..\models\best_openvino_model" True 1280 0.1 0.2 "NPU"

call conda deactivate