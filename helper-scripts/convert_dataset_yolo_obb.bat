@REM if (%CONDA_DEFAULT_ENV% == "label-backend")
call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: yolo to obb
@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_5.yaml"

:: obb to yolo
@REM call python  tools\build_dataset.py --obb-to-yolo --data-config-yaml "data\dataset_4.yaml"

call conda deactivate
