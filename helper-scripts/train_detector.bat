call .\activate_label-backend_env.bat

call cd ..

call wandb online

:: freeze layers for transfer-learning
:: --freeze 22
::--mlflow-model-alias "last"^

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0001 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml"^
@REM     --path-weights "runs\mlflow\640061168132957373\60639d408efd475d97989b8542456d8d\artifacts\weights\best.pt"^
@REM     --run-name "yolov10s"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-v3"^ 

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 20 --lr0 0.00005 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "base_models_weights\yolov5mu.pt" ^
@REM     --run-name "yolov5mu"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-v3"^ 
   
@REM :: Transfer learning from a model registered on mlflow
@REM call python tools\cli.py --start-training --batchsize 128  --epochs 20 --lr0 0.0001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005 --freeze 21^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --run-name "detector" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4" "transferlearning"^ 

@REM :: Transfer learning from a pth file
@REM call python tools\cli.py --start-training --batchsize 128  --epochs 20 --lr0 0.0001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005 --freeze 23^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/095bc5dc29c74399b19e8f4c60fa4fa1/artifacts/weights/best.pt" ^
@REM     --run-name "yolov5mu" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4" "transferlearning"^

:: Finetuning from a pth file
call python tools\cli.py --start-training --batchsize 64  --epochs 25 --lr0 0.001 --lrf 0.01 --patience 5 --is-detector^
    --scale 0.0 --weight-decay 0.0005^
    --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
    --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/095bc5dc29c74399b19e8f4c60fa4fa1/artifacts/weights/best.pt" ^
    --run-name "yolov5mu" --project-name "wildAI-detection"^
    --tag "dataset-v4"^


call conda deactivate

:: uncomment to disable immediate shutdown
@REM shutdown -s