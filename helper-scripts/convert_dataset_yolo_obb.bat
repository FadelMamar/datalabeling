@REM if (%CONDA_DEFAULT_ENV% == "label-backend")
@REM call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
@REM call conda activate label-backend

call cd ..

:: yolo to obb
call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\data_config.yaml"

:: obb to yolo
call python  tools\build_dataset.py --obb-to-yolo --data-config-yaml "data\data_config.yaml"
