call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools"

:: add --clear-yolo-dir to clear data in ./data/train

call python cli.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 1 --is-detector^
    --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train\labels" ^
    --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train\images" ^
    --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
    --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json" ^
    --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml"

call conda deactivate