call .\activate_label-backend_env.bat

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: set to 'offline' when having issues with internet, otherwise use 'online'
call wandb offline

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

call python tools\cli.py --start-training --batchsize 8  --weight-decay 0.0005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 20 --is-detector ^
    --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
    --height 800 --width 800^
    --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/3615defe14514a00b97ef756a815bc44/artifacts/weights/best.pt" ^
    --run-name "yolov8_obb-CL-HN" --project-name "wildAI-detection"^
    --tag "CL" "HN" ^
    --ptr-data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_pretraining.yaml"  ^
    --ptr-tilesize 640 --ptr-epochs 15^
    --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
    --cl-data-config-yaml "data\dataset_identification.yaml" ^
    --cl-ratios 0 1 2.5 7.5 10 ^
    --cl-epochs 10 15 15 20 20 ^
    --cl-freeze 0 0 10 15 15 ^
    --cl-lr0s 0.0001 0.0001 0.00005 0.00005 0.00005 ^
    --hn-save-dir "D:\PhD\Data per camp\IdentificationDataset\hard_samples" --use-hn-learning ^
    --hn-num-epochs 7 --hn-data-config-yaml "data\dataset_identification.yaml" ^
    --hn-is-yolo-obb


call conda deactivate