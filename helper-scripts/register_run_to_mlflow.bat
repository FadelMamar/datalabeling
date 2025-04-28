call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

@REM --use-sliding-window adding this flag will enabled sahi inference


@REM call uv run tools\register_model.py --exp-name "labeler" --model ""^
@REM         --model-name "obb-detector" --mlflow-tracking-uri "http://localhost:5000"^
@REM         --tilesize 800 --imgsz 800 --confidence-threshold 0.1 --overlap-ratio 0.15^
@REM         --is-yolo-obb --use-sliding-window

@REM call uv run tools\register_model.py --exp-name "labeler" --model ""^
@REM         --model-name "detector" --mlflow-tracking-uri "http://localhost:5000"^
@REM         --tilesize 800 --imgsz 800 --confidence-threshold 0.1 --overlap-ratio 0.15^
@REM         --is-yolo-obb --use-sliding-window


call conda deactivate
