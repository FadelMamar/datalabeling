call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools"

:: add --clear-yolo-dir to clear data in ./data/train
:: --load-coco-annotations

:: builds detection dataset
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.1 --is-detector^
@REM     --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train-wildai\labels" ^
@REM     --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train-wildai\images" ^
@REM     --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
@REM     --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json" ^

:: **Saves only empty tiles**
@REM call python build_dataset.py --build-yolo-dataset  --is-detector --clear-yolo-dir^
@REM     --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\negative-samples\labels" ^
@REM     --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\negative-samples\images" ^
@REM     --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
@REM     --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json" ^
@REM     --save-only-empty --empty-ratio 0.1


call conda deactivate