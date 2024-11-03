call .\activate_label-backend_env.bat

call cd ..\tools

:: add --clear-yolo-dir to clear data in ./data/train
:: --load-coco-annotations

:: Example builds detection dataset
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.1 --is-detector^
@REM     --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train-wildai\labels" ^
@REM     --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train-wildai\images" ^
@REM     --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
@REM     --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json"

:: Example **Saves only empty tiles**
@REM call python build_dataset.py --build-yolo-dataset  --is-detector --clear-yolo-dir --empty-ratio 0.1^
@REM     --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\negative-samples\labels" ^
@REM     --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\negative-samples\images" ^
@REM     --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
@REM     --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json" ^
@REM     --save-only-empty 

:: Rep 1 data
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.0 --is-detector --min-visibility 0.8^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1\train\labels" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1\train\images" ^
@REM     --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\train\coco-format" ^
@REM     --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\train\labelstudio"

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.0 --is-detector --min-visibility 0.8^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1\val\labels" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1\val\images" ^
@REM     --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\val\coco-format" ^
@REM     --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\val\labelstudio"

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.0 --is-detector --min-visibility 0.8^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1\test\labels" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1\test\images" ^
@REM     --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\test\coco-format" ^
@REM     --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\test\labelstudio"

:: Rep 2 data
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.0 --is-detector --min-visibility 0.8^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 2\train\labels" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 2\train\images" ^
@REM     --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\train\coco-format" ^
@REM     --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\train\labelstudio"

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.0 --is-detector --min-visibility 0.8^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 2\val\labels" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 2\val\images" ^
@REM     --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\val\coco-format" ^
@REM     --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\val\labelstudio"

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.0 --is-detector --min-visibility 0.8^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 2\test\labels" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 2\test\images" ^
@REM     --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\test\coco-format" ^
@REM     --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\test\labelstudio"

call conda deactivate