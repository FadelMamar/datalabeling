call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: add --clear-yolo-dir to clear data in ./data/train
:: --dest-path-labels
:: --dest-path-images
:: --label-map

@REM :: TRAIN
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 10. --parse-ls-config --label-map "exported_annotations\label_mapping.json" --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\train\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\train\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\train\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\train\coco-format" ^
@REM     --height 1280 --width 1280 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^
@REM     --discard-labels "other" "rocks" "vegetation" "detection" "termite mound" "label" 
    


:: VAL
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 20. --parse-ls-config --label-map "D:\PhD\Data per camp\IdentificationDataset\label_mapping.json" --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\val\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\val\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\val\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\val\coco-format" ^
@REM     --height 1280 --width 1280 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^
@REM     --discard-labels "other" "rocks" "vegetation" "detection" "termite mound" "label"
    

@REM :: TEST
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 20. --parse-ls-config --label-map "D:\PhD\Data per camp\IdentificationDataset\label_mapping.json" --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\test\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\test\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\test\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\test\coco-format" ^
@REM     --height 1280 --width 1280 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^ 
@REM --discard-labels "other" "rocks" "vegetation" "detection" "termite mound" "label"


call conda deactivate