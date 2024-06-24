call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\src"

:: add --clear-yolo-dir to clear data in ./data/train

call python cli.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 1

call conda deactivate