call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

@REM --use-sliding-window adding this flag will enabled sahi inference


call uv run tools\register_model.py --exp-name "labeler" --model ""^
        --model-name "obb-detector" --mlflow-tracking-uri "http://localhost:5000"^
        --tilesize 1280 --imgsz 1280 --confidence-threshold 0.1 --overlap-ratio 0.15^
        --is-yolo-obb --use-sliding-window

call conda deactivate