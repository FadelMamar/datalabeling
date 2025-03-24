
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

 
@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\configs\yolo_configs\dataset_identification-detection.yaml" --skip


:: set to 'offline' when having issues with internet, otherwise use 'online'
@REM # --augment enables TestTime augmentation

call uv run tools\validate.py --splits "val" --is-detector --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
    --weights "runs/mlflow/140168774036374062/3964086fef714345bd681e0f4f366614/artifacts/weights/best.pt"^
    --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.25 --name "yolo11-obb-X#mlflow-3964086fef714345bd681e0f4f366614" --batch-size 32 --max-det 100 ^
    --project-name "wildai-validation" --save-txt --plots


call uv run tools\validate.py --splits "val" --is-detector --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
    --weights "runs/mlflow/140168774036374062/8c6c27095ab7493b95e1ec46c03b7d1b/artifacts/weights/best.ptt"^
    --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.25 --name "yolo11-obb-X#mlflow-8c6c27095ab7493b95e1ec46c03b7d1b" --batch-size 32 --max-det 100 ^
    --project-name "wildai-validation" --save-txt --plots