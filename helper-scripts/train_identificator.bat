
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat


:: set to 'offline' when having issues with internet, otherwise use 'online'
call wandb online

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



@REM herdnet

@REM call uv run tools\train-herdnet.py --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --run-name "herdnet-Identif" --lr0 0.0001 --batchsize 16 ^
@REM                                    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth" ^
@REM                                    --imgsz 800 --cl-ratios 0.0 --project-name "Herdnet" --epochs 30 --device "cuda"

@REM call uv run tools\train-herdnet.py --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --run-name "herdnet-Det" --lr0 0.0001 --batchsize 16 ^
@REM                                     --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"  ^
@REM                                     --imgsz 800 --cl-ratios 0.0 --project-name "Herdnet" --epochs 30 --device "cuda"

call uv run tools\train-herdnet.py --data-config-yaml "configs\yolo_configs\dataset_pretraining.yaml" --run-name "herdnet-Det-PTR" --lr0 0.0001 --batchsize 16 ^
                                    --path-weights "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\20220329_HerdNet_Ennedi_dataset_2023.pth"  ^
                                    --imgsz 640 --cl-ratios 0.0 --project-name "Herdnet" --epochs 30 --device "cuda"

@REM shutdown -s


@REM  Identificator

@REM @REM Convert datasets to yolo
@REM call uv run  tools\build_dataset.py --yolo-to-obb --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --skip
@REM call uv run  tools\build_dataset.py --obb-to-yolo --data-config-yaml "data\dataset_pretraining.yaml" --skip

@REM call uv run tools\cli.py --start-training --batchsize 8  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 10 ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 0 ^
@REM     --height 800 --width 800 ^
@REM     --path-weights "wildAI-detection\yolov12x-Det-CL_emptyRatio_7.5_freeze_15\weights\best.pt" ^
@REM     --run-name "yolov12x-Identif" --project-name "wildAI-detection"^
@REM     --tag "CL" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\DetectionDataset\continuous_learning" ^
@REM     --cl-data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --use-continual-learning ^
@REM     --cl-ratios 2.5 ^
@REM     --cl-epochs 15  ^
@REM     --cl-freeze 15 ^
@REM     --cl-lr0s 0.0001 ^
@REM     --cl-batch-size 16

@REM call uv run  tools\build_dataset.py --obb-to-yolo --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --skip
@REM call uv run  tools\build_dataset.py --obb-to-yolo --data-config-yaml "configs\yolo_configs\dataset_pretraining.yaml" --skip

@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800^
@REM     --path-weights "runs/mlflow/140168774036374062/e5e3bf93d34f48f1bb7d0a648530bb45/artifacts/weights/best.pt" ^
@REM     --run-name "yolov5s-CL-PTR-Identification" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_identification.yaml" ^
@REM     --cl-ratios 0 2.5 7.5 ^
@REM     --cl-epochs 15 10 10 ^
@REM     --cl-freeze 0 5 10  ^
@REM     --cl-lr0s 0.0001 0.0001 0.0001 ^

@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 1 ^
@REM     --height 800 --width 800^
@REM     --path-weights "..." ^
@REM     --run-name "yolov12s-Identification" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_identification.yaml" ^
@REM     --cl-ratios 2.5  ^
@REM     --cl-freeze 15   ^
@REM     --cl-lr0s 0.00005 

@REM @REM @REM convert datasets to yolo-obb
@REM call uv run  tools\build_dataset.py --yolo-to-obb --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --skip
@REM call uv run  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_pretraining.yaml" --skip

@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800^
@REM     --path-weights "runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt" ^
@REM     --run-name "yolov11s_obb-CL-PTR-Identification" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_identification.yaml" ^
@REM     --cl-ratios 0 2.5 7.5 ^
@REM     --cl-epochs 15 10 10 ^
@REM     --cl-freeze 0 5 10  ^
@REM     --cl-lr0s 0.0001 0.0001 0.0001 ^


@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800^
@REM     --path-weights "runs/mlflow/140168774036374062/8a76c60253fc48788b5324096d035420/artifacts/weights/best.pt" ^
@REM     --run-name "yolov8s_obb-CL-PTR-Identification" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_identification.yaml" ^
@REM     --cl-ratios 0 2.5 7.5 ^
@REM     --cl-epochs 15 10 10 ^
@REM     --cl-freeze 0 5 10  ^
@REM     --cl-lr0s 0.0001 0.0001 0.0001 ^

@REM call uv run tools\cli.py --start-training --batchsize 8  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 10 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800 ^
@REM     --path-weights "runs/mlflow/140168774036374062/3964086fef714345bd681e0f4f366614/artifacts/weights/last.pt"^
@REM     --run-name "yolov11x-CL-Detector" --project-name "wildAI-detection"^
@REM     --tag "CL" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\DetectionDataset\continuous_learning" ^
@REM     --cl-data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --use-continual-learning ^
@REM     --cl-ratios 2.5 7.5 ^
@REM     --cl-epochs 10 10 ^
@REM     --cl-freeze 15 20 ^
@REM     --cl-lr0s 0.00005 0.00005 ^
@REM     --cl-batch-size 16

@REM  DETECTOR


@REM @REM Convert datasets to yolo
@REM call uv run  tools\build_dataset.py --obb-to-yolo --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --skip
@REM call uv run  tools\build_dataset.py --obb-to-yolo --data-config-yaml "configs\yolo_configs\dataset_pretraining.yaml" --skip


@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800 ^
@REM     --path-weights "base_models_weights\yolov5su.pt" ^
@REM     --run-name "yolov5s_PTR-CL-Detector" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --ptr-data-config-yaml "data\dataset_pretraining.yaml"  ^
@REM     --ptr-tilesize 640 --ptr-epochs 15 --use-pretraining ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_identification-detection.yaml" ^
@REM     --cl-ratios 0 1 2.5 7.5 ^
@REM     --cl-epochs 20 20 10 7 ^
@REM     --cl-freeze 0 0 10 18  ^
@REM     --cl-lr0s 0.0001 0.0001 0.00005 0.00005 ^

@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 0 --val "True" ^
@REM     --height 800 --width 800^
@REM     --path-weights "wildAI-detection\yolov12x-Det-CL_emptyRatio_0.0_freeze_None\weights\best.pt" ^
@REM     --run-name "yolov12x-Det" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --ptr-data-config-yaml "configs\yolo_configs\dataset_pretraining.yaml"  ^
@REM     --ptr-tilesize 640 --ptr-epochs 10 --ptr-batchsize 8 ^
@REM     --cl-save-dir "D:\PhD\Data per camp\DetectionDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --cl-batch-size 8 ^
@REM     --cl-ratios 2.5 7.5 ^
@REM     --cl-epochs 10 5 ^
@REM     --cl-freeze 15 20  ^
@REM     --cl-lr0s 0.00005 0.00005



@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.0005 --optimizer "auto" --optimizer-momentum 0.99 --lr0 0.0003 --lrf 0.01 --patience 10 --is-detector ^
@REM     --scale 0.0 --mosaic 0.0 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 0 ^
@REM     --height 800 --width 800^
@REM     --path-weights "base_models_weights\yolov9c-seg.pt" --task "segment" ^
@REM     --run-name "yolov9c-seg" --project-name "wildAI-detection"^
@REM     --tag "CL" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --cl-batch-size 16^
@REM     --cl-data-config-yaml "configs\yolo_configs\dataset_identification-seg.yaml" --use-continual-learning ^
@REM     --cl-ratios 0.1 ^
@REM     --cl-epochs 30 ^
@REM     --cl-freeze 0 ^
@REM     --cl-lr0s 0.0003


@REM @REM convert datasets to yolo-obb
@REM call uv run  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_identification-detection.yaml" --skip
@REM call uv run  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_pretraining.yaml" --skip

@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800 ^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/2ff9bb7a991c4cd1a6eabfff0f73386d/artifacts/weights/last.pt" ^
@REM     --run-name "yolov11_obb-PTR-CL-Detector" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --ptr-data-config-yaml "data\dataset_pretraining.yaml"  ^
@REM     --ptr-tilesize 640 --ptr-epochs 15 ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_identification-detection.yaml" --cl-batch-size 24 ^
@REM     --cl-ratios 0 1 2.5 7.5 ^
@REM     --cl-epochs 20 20 10 7 ^
@REM     --cl-freeze 0 0 10 18  ^
@REM     --cl-lr0s 0.0001 0.0001 0.00005 0.00005 ^


@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800 ^
@REM     --path-weights "runs/mlflow/140168774036374062/3615defe14514a00b97ef756a815bc44/artifacts/weights/best.pt" ^
@REM     --run-name "yolov8s_obb-PTR-CL-Detector" --project-name "wildAI-detection"^
@REM     --tag "CL" "PTR" ^
@REM     --ptr-data-config-yaml "data\dataset_pretraining.yaml"  ^
@REM     --ptr-tilesize 640 --ptr-epochs 15 ^
@REM     --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --cl-batch-size 24^
@REM     --cl-data-config-yaml "data\dataset_identification-detection.yaml" --use-continual-learning ^
@REM     --cl-ratios 0 1 2.5 7.5 ^
@REM     --cl-epochs 20 20 10 7 ^
@REM     --cl-freeze 0 0 10 18  ^
@REM     --cl-lr0s 0.0001 0.0001 0.00005 0.00005 ^



@REM call deactivate

@REM shutdown -s