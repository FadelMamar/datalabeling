call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\src"

:: add --clear-yolo-dir to clear data in ./data/train
:: --dest-path-labels
:: --dest-path-images
:: add --is-detector to train an animal detector

call python cli.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 1 ^
    --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\train\labels" --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\train\images" ^
    --coco-json-dir "D:\PhD\Data per camp\IdentificationDataset\train\coco_json" --ls-json-dir "D:\PhD\Data per camp\IdentificationDataset\train\labelstudio_json" ^
    --data-config-yaml "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml" --label-map "D:\PhD\Data per camp\IdentificationDataset\label_mapping.json"

call python cli.py --build-yolo-dataset --clear-yolo-dir --empty-ratio 1 ^
    --dest-path-labels "D:\PhD\Data per camp\IdentificationDataset\val\labels" --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\val\images" ^
    --coco-json-dir "D:\PhD\Data per camp\IdentificationDataset\val\coco_json" --ls-json-dir "D:\PhD\Data per camp\IdentificationDataset\val\labelstudio_json" ^
    --data-config-yaml "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml" --label-map "D:\PhD\Data per camp\IdentificationDataset\label_mapping.json"

call conda deactivate