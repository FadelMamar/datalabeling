call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: --cos-lr 
call wandb online

call python ".\src\train.py" --batchsize 32  --epochs 100 --lr0 0.001 --patience 50^
    --data-config-yaml "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml" ^
    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8.kaza.pt" ^
    --run-name "identificator" ^
    --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\train\images"

call python ".\src\train.py" --batchsize 32  --epochs 100 --lr0 0.001 --patience 50^
    --data-config-yaml "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml" ^
    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8m.pt" ^
    --run-name "identificator" ^
    --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\train\images"

call python ".\src\train.py" --batchsize 32  --epochs 100 --lr0 0.001 --patience 50^
    --data-config-yaml "D:\PhD\Data per camp\IdentificationDataset\data_config.yaml" ^
    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8s.pt" ^
    --run-name "identificator" ^
    --dest-path-images "D:\PhD\Data per camp\IdentificationDataset\train\images"


call conda deactivate