call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

:: add --clear-yolo-dir to clear data in ./data/train

call python .\src\cli.py --build-yolo-dataset --clear-yolo-dir

call conda deactivate