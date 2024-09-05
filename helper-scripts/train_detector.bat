call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call wandb online

:: freeze layers for transfer-learning
:: --freeze 22

call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.00001 --lrf 0.01 --patience 10 --is-detector^
    --scale 0.0 --weight-decay 0.0005^
    --mlflow-model-alias "last"^
    --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8.kaza.pt" ^
    --run-name "detector" --project-name "wildAI-detection"^
    --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train\images"^
    --tag "dataset-v3"^ 
   
call python tools\cli.py --start-training --batchsize 64  --epochs 50 --lr0 0.00001 --lrf 0.01 --patience 10 --is-detector^
    --scale 0.0 --weight-decay 0.0005 --freeze 22^
    --mlflow-model-alias "last"^
    --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8.kaza.pt" ^
    --run-name "detector" --project-name "wildAI-detection"^
    --dest-path-images "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\train\images"^
    --tag "dataset-v3" "transferlearning"^ 


call conda deactivate