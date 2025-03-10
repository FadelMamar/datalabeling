call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: add --clear-yolo-dir to clear data in ./data/train
:: --dest-path-labels
:: --dest-path-images
:: --label-map

@REM   0: Waterbuck
@REM   1: buffalo
@REM   2: bushbuck
@REM   3: duiker
@REM   4: giraffe
@REM   5: impala
@REM   6: kudu
@REM   7: nyala
@REM   8: nyala(m)
@REM   9: other animal
@REM   10: reedbuck
@REM   11: roan
@REM   12: sable
@REM   13: warthog
@REM   14: wildebeest
@REM   15: zebra
@REM   16: colour impala
@REM   17: lechwe
@REM   18: Tsessebe

@REM IDENTIFICATION

@REM :: TRAIN
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 7.5 --parse-ls-config --label-map "exported_annotations\label_mapping.json" --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\train\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\train\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\train\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\train\coco-format" ^
@REM     --height 800 --width 800 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^
@REM     --keep-labels "buffalo" "impala" "nyala" "nyala(m)" "roan" "sable"

@REM :: VAL
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 15. --parse-ls-config --label-map "exported_annotations\label_mapping.json" --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\val\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\val\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\val\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\val\coco-format" ^
@REM     --height 800 --width 800 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^
@REM     --keep-labels "buffalo" "impala" "nyala" "nyala(m)" "roan" "sable"

@REM :: TEST
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 20. --parse-ls-config --label-map "exported_annotations\label_mapping.json" --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\test\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\test\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\test\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\test\coco-format" ^
@REM     --keep-labels "buffalo" "impala" "nyala" "nyala(m)" "roan" "sable" ^
@REM     --height 800 --width 800 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    

@REM DETECTION
@REM :: TRAIN
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 7.5 --parse-ls-config --label-map "exported_annotations\label_mapping.json" --is-detector --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Identification-split\train\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\train\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Identification-split\train\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\train\coco-format" ^
@REM     --height 800 --width 800 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^
@REM     --discard-labels "other" "rocks" "vegetation" "detection" "termite mound" "label" "wildlife"

@REM :: VAL
@REM call python  tools\build_dataset.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 15. --parse-ls-config --label-map "exported_annotations\label_mapping.json" --is-detector --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Identification-split\val\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\val\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Identification-split\val\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\val\coco-format" ^
@REM     --height 800 --width 800 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml" ^
@REM     --discard-labels "other" "rocks" "vegetation" "detection" "termite mound" "label" "wildlife"
    

@REM :: TEST
@REM call python  tools\build_dataset.py --build-yolo-dataset --load-coco-annotations --empty-ratio 20. --parse-ls-config --label-map "exported_annotations\label_mapping.json" --is-detector --min-visibility 0.7 ^
@REM     --dest-path-labels "D:\PhD\Data per camp\DetectionDataset\Identification-split\test\labels"  --ls-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\test\labelstudio" ^
@REM     --dest-path-images "D:\PhD\Data per camp\DetectionDataset\Identification-split\test\images" --coco-json-dir "D:\PhD\Data per camp\Exported annotations and labels\identification-splits\test\coco-format" ^
@REM     --discard-labels "other" "rocks" "vegetation" "detection" "termite mound" "label" "wildlife" ^
@REM     --height 800 --width 800 --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_identification.yaml"
    



call deactivate