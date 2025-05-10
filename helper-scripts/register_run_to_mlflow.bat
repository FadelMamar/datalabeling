call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

@REM --use-sliding-window adding this flag will enabled sahi inference

call set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set AWS_ACCESS_KEY_ID=minioadmin

@REM call uv run tools\register_model.py --exp-name "labeler" --model "D:\datalabeling\base_models_weights\best.pt"^
@REM         --model-name "labeler" --mlflow-tracking-uri "http://localhost:5000"^
@REM         --tilesize 800 --imgsz 800 --confidence-threshold 0.1 --overlap-ratio 0.15^
@REM         --use-sliding-window

@REM call uv run tools\register_model.py --exp-name "labeler" --model ""^
@REM         --model-name "detector" --mlflow-tracking-uri "http://localhost:5000"^
@REM         --tilesize 800 --imgsz 800 --confidence-threshold 0.1 --overlap-ratio 0.15^
@REM         --use-sliding-window


call deactivate
