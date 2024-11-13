call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: set to 'offline' when having issues with internet, otherwise use 'online'
call wandb offline

@REM :: freeze layers for transfer-learning
@REM :: --freeze 22 @REM ::--mlflow-model-alias "last"^

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_0-1.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8.kaza.pt"^
@REM     --run-name "yolov8.kaza"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-0-1" 

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_0-2.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8.kaza.pt"^
@REM     --run-name "yolov8.kaza"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-0-2"

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 5 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_1.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8m-obb.pt"^
@REM     --run-name "yolov8m-obb"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-1"


@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_1.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov5mu.pt" ^
@REM     --run-name "yolov5mu"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-1"

@REM call python tools\cli.py --start-training --batchsize 24  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_5.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8m-obb.pt"^
@REM     --run-name "yolov8m-obb"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-5"

@REM call python tools\cli.py --start-training --batchsize 64  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM      --scale 0.0 --weight-decay 0.0005^
@REM      --data-config-yaml "data\dataset_5.yaml"^
@REM      --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8s-obb.pt"^
@REM      --run-name "yolov8s-obb"^
@REM      --project-name "wildAI-detection"^
@REM      --tag "dataset-2"

 call python tools\cli.py --start-training --batchsize 64  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
     --scale 0.0 --weight-decay 0.0005^
     --data-config-yaml "data\dataset_2.yaml"^
     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolo11s-obb.pt"^
     --run-name "yolo11s-obb"^
     --project-name "wildAI-detection"^
     --tag "dataset-2"

@REM call python tools\cli.py --start-training --batchsize 64  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_2.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8n-obb.pt"^
@REM     --run-name "yolov8n-obb"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-2"

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_2.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov5mu.pt" ^
@REM     --run-name "yolov5mu"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-2"

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_3.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8m-obb.pt"^
@REM     --run-name "yolov8m-obb"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-3"

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_3.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov5mu.pt" ^
@REM     --run-name "yolov5mu"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-3"

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "data\dataset_4.yaml"^
@REM     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolov8m-obb.pt"^
@REM     --run-name "yolov8m-obb"^
@REM     --project-name "wildAI-detection"^
@REM     --tag "dataset-4"


@REM :: obb
@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.0001 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "base_models_weights\yolov8m-obb.pt" ^
@REM     --run-name "yolov8m-obb" --project-name "wildAI-detection"^
@REM     --tag "dataset-v3"^

@REM call python tools\cli.py --start-training --batchsize 32  --epochs 50 --lr0 0.001 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "yolov8m-obb.pt" ^
@REM     --run-name "yolov8m-obb" --project-name "wildAI-detection"^
@REM     --tag "dataset-v3"^

@REM :: Transfer learning from a model registered on mlflow
@REM call python tools\cli.py --start-training --batchsize 128  --epochs 20 --lr0 0.0001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005 --freeze 21^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --run-name "detector" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4" "transferlearning"^ 

@REM :: Finetuning from a model registered on mlflow
@REM call python tools\cli.py --start-training --batchsize 32  --epochs 20 --lr0 0.0001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --run-name "detector" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4" ^ 

@REM :: Transfer learning from a pth file
@REM call python tools\cli.py --start-training --batchsize 128  --epochs 20 --lr0 0.001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005 --freeze 23^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/095bc5dc29c74399b19e8f4c60fa4fa1/artifacts/weights/best.pt" ^
@REM     --run-name "yolov5mu" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4" "transferlearning"^



@REM :: Finetuning from a pth file
@REM call python tools\cli.py --start-training --batchsize 64  --epochs 25 --lr0 0.001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/095bc5dc29c74399b19e8f4c60fa4fa1/artifacts/weights/best.pt" ^
@REM     --run-name "yolov5mu" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4"^




call conda deactivate

:: uncomment to disable immediate shutdown
@REM shutdown -s