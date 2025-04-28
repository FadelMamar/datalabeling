call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call helper-scripts\build_yolo_seg_dataset.bat

call helper-scripts\train_identificator.bat

call uv run tools\train-herdnet.py

@REM shutdown -s
