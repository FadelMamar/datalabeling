call .\activate_label-backend_env.bat

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

@REM  call python tools\cli.py --start-training --batchsize 64  --epochs 50 --lr0 0.0005 --lrf 0.01 --patience 20 --is-detector^
@REM      --scale 0.0 --weight-decay 0.0005^
@REM      --data-config-yaml "data\dataset_2.yaml"^
@REM      --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolo11s-obb.pt"^
@REM      --run-name "yolo11s-obb"^
@REM      --project-name "wildAI-detection"^
@REM      --tag "dataset-2"

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

@REM    use_continual_learning:bool=False
@REM     cl_ratios:Sequence[float]=(1.0,)
@REM     cl_epochs:Sequence[int]=(20,)
@REM     cl_freeze:Sequence[int]=(None,)
@REM     cl_lr0s:Sequence[float]=(1e-5,)
@REM     cl_save_dir:str=None # should be given!
@REM    cl_data_config_yaml:str=None # should be given!

@REM    use_hn_learning:bool=False, Default params below
@REM hn_save_dir:str=None
@REM hn_data_config_yaml:str=None
@REM hn_num_epochs:int=10
@REM hn_freeze:int=20
@REM hn_lr0:float=1e-4
@REM hn_lrf:float=1e-2
@REM hn_batch_size:int=32
@REM hn_is_yolo_obb:bool=False
@REM hn_use_sliding_window:bool=True
@REM hn_overlap_ratio:float=0.2
@REM hn_map_thrs:float=0.35 # mAP threshold. lower than it is considered sample of interest
@REM hn_score_thrs:float=0.7
@REM hn_confidence_threshold:float=0.1
@REM hn_ratio:int=20 # ratio = num_empty/num_non_empty. Higher allows to look at all saved empty images
@REM hn_uncertainty_thrs:float=5 # helps to select those with high uncertainty
@REM hn_uncertainty_method:str="entropy"

@REM :: obb

call wandb offline

@REM call python tools\cli.py --start-training --batchsize 16  --weight-decay 0.0005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 10 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0^
@REM     --height 1280 --width 1280^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/1ae90d2390a944a8b1a05540e9daaa72/artifacts/weights/best.pt" ^
@REM     --run-name "yolov8s_obb-PTR-CL-HN" --project-name "wildAI-detection"^
@REM     --tag "PTR" "CL" "HN" ^
@REM     --ptr-data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_pretraining.yaml" --use-pretraining ^
@REM     --ptr-tilesize 640 --ptr-epochs 5^
@REM     --cl-save-dir "D:\PhD\Data per camp\DetectionDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_cl.yaml" ^
@REM     --cl-ratios 0 0.5 1 2.5 7.5 10 ^
@REM     --cl-epochs 10 10 10 5 5 20 ^
@REM     --cl-freeze 0 0 5 10 15 20 ^
@REM     --cl-lr0s 0.0001 0.0001 0.0001 0.0001 0.0001 0.00001 ^
@REM     --hn-save-dir "D:\PhD\Data per camp\DetectionDataset\hard_samples" --use-hn-learning ^
@REM     --hn-num-epochs 10 --hn-data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_hn.yaml" ^
@REM     --hn-is-yolo-obb --hn-use-sliding_window

call python tools\cli.py --start-training --batchsize 16  --weight-decay 0.0005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 10 --is-detector ^
    --scale 0.5 --mosaic 0.2 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0^
    --height 1280 --width 1280^
    --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/57daf3bcd99b4dd4b040cb4f8670960c/artifacts/weights/best.pt" ^
    --run-name "yolov8s_obb-PTR-CL-HN" --project-name "wildAI-detection"^
    --tag "HN" ^
    --hn-save-dir "D:\PhD\Data per camp\DetectionDataset\hard_samples" --use-hn-learning ^
    --hn-num-epochs 5 --hn-freeze 22 --hn-data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_hn.yaml" ^
    --hn-tilesize 1280 --hn-imgsz 1280 --hn-lr0 0.0001 --hn-lrf 0.01 ^
    --hn-is-yolo-obb --hn-load-results


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

:: Transfer learning from a model registered on mlflow
@REM call python tools\cli.py --start-training --batchsize 16  --epochs 20 --lr0 0.0001 --weight-decay 0.05 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.5 --mosaic 0.3 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0^
@REM     --height 640 --width 640^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "D:\PhD\Data per camp\DetectionDataset\hard_samples\hard_samples.yaml"^
@REM     --run-name "obb-detector" --project-name "wildAI-detection"^
@REM     --tag "hard_samples" "transfer-learning"

@REM call python tools\cli.py --start-training --batchsize 16  --epochs 20 --lr0 0.0001 --weight-decay 0.05 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.5 --mosaic 0.3 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0^
@REM     --height 640 --width 640^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "D:\PhD\Data per camp\DetectionDataset\hard_samples\hard_samples.yaml"^
@REM     --run-name "obb-detector" --project-name "wildAI-detection"^
@REM     --tag "hard_samples" "transfer-learning"

@REM call python tools\cli.py --start-training --batchsize 16  --epochs 20 --lr0 0.0001 --weight-decay 0.05 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.8 --mosaic 1. --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.3^
@REM     --height 640 --width 640^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "D:\PhD\Data per camp\DetectionDataset\hard_samples\hard_samples.yaml"^
@REM     --run-name "obb-detector" --project-name "wildAI-detection"^
@REM     --tag "hard_samples" "transfer-learning"

@REM call python tools\cli.py --start-training --batchsize 16  --epochs 20 --lr0 0.0001 --weight-decay 0.05 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.8 --mosaic 1. --copy-paste 0.5 --mixup 0.0 --rotation-degree 45. --erasing 0.3^
@REM     --height 640 --width 640^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "D:\PhD\Data per camp\DetectionDataset\hard_samples\hard_samples.yaml"^
@REM     --run-name "obb-detector" --project-name "wildAI-detection"^
@REM     --tag "hard_samples" "transfer-learning"

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