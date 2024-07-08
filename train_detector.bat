call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"


call wandb online

:: freeze layers for transfer-learning

call python ".\src\train.py" --batchsize 32  --epochs 50 --lr0 0.0001 --lrf 0.1 --patience 10 --is-detector^
    --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8.kaza.pt" ^
    --run-name "detector" --project-name "wildAI-detection"^
    --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train\images"^
    --tag "dataset-v2"^ 
    --weightdecay 0.0005
:: --freeze 22


call conda deactivate