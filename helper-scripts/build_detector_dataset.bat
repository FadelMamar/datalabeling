call .\activate_label-backend_env.bat

call cd ..\tools

@REM :: add --clear-yolo-dir to clear data in ./data/train
@REM :: --load-coco-annotations

@REM :: Example builds detection dataset
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0.1 --is-detector^
@REM     --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train-wildai\labels" ^
@REM     --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train-wildai\images" ^
@REM     --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
@REM     --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json"

@REM :: Example **Saves only empty tiles**
@REM call python build_dataset.py --build-yolo-dataset  --is-detector --clear-yolo-dir --empty-ratio 0.1^
@REM     --dest-path-labels "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\negative-samples\labels" ^
@REM     --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\negative-samples\images" ^
@REM     --coco-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\coco-format" ^
@REM     --ls-json-dir "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\exported_annotations\json" ^
@REM     --save-only-empty 

@REM :: Rep 1 dry data
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10. --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1\train\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\train\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1\train\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\train\labelstudio"  ^
@REM     --height 1280 --width 1280 

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 20. --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1\val\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\val\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1\val\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\val\labelstudio"  ^
@REM     --height 1280 --width 1280 

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0. --save-all --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1\test\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\test\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1\test\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\test\labelstudio"  ^
@REM     --height 1280 --width 1280

:: Rep 1 wet
call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10.  --parse-ls-config --is-detector --min-visibility 0.8 ^
    --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1 - Wet\train\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Wet season - Rep 1\train\coco-format" ^
    --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1 - Wet\train\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Wet season - Rep 1\train\labelstudio"  ^
    --height 1280 --width 1280

call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10.  --parse-ls-config --is-detector --min-visibility 0.8 ^
    --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1 - Wet\val\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Wet season - Rep 1\val\coco-format" ^
    --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1 - Wet\val\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Wet season - Rep 1\val\labelstudio"  ^
    --height 1280 --width 1280

call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0. --save-all --parse-ls-config --is-detector --min-visibility 0.8 ^
    --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 1 - Wet\test\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Wet season - Rep 1\test\coco-format" ^
    --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 1 - Wet\test\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Wet season - Rep 1\test\labelstudio"  ^
    --height 1280 --width 1280


@REM :: Rep 2 dry data
@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10. --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 2\train\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\train\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 2\train\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\train\labelstudio"  ^
@REM     --height 1280 --width 1280 

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 20. --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 2\val\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\val\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 2\val\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\val\labelstudio"  ^
@REM     --height 1280 --width 1280 

call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0. --save-all --parse-ls-config --is-detector --min-visibility 0.8 ^
    --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 2\test\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\test\coco-format" ^
    --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 2\test\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 2\test\labelstudio"  ^
    --height 1280 --width 1280


@REM :: Rep 3dry  data
  

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10. --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 3\train\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 3\train\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 3\train\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 3\train\labelstudio"  ^
@REM     --height 1280 --width 1280 

@REM call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 20. --parse-ls-config --is-detector --min-visibility 0.8 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 3\val\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 3\val\coco-format" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 3\val\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 3\val\labelstudio"  ^
@REM     --height 1280 --width 1280

call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 0. --save-all --parse-ls-config --is-detector --min-visibility 0.8 ^
    --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Rep 3\test\labels" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 3\test\coco-format" ^
    --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Rep 3\test\images"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 3\test\labelstudio"  ^
    --height 1280 --width 1280

@REM :: Bushriver data
call python  build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10. --parse-ls-config --is-detector --min-visibility 0.8^
    --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\bushriver\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\bushriver-extra\labelstudio" ^
    --dest-path-images "D:\PhD\Data per camp\DetectionDataset\bushriver\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\Dry season - Rep 1\bushriver-extra\coco-format" ^



call conda deactivate

shutdown -s