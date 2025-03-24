
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat


@REM Ultralytics and HerdNet

@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\configs\yolo_configs\dataset_identification-detection.yaml" --skip


@REM  YOLO-obb
@REM # --augment enables TestTime augmentation

call uv run tools\validate.py --splits "test" --is-detector --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
    --weights "runs/mlflow/140168774036374062/685cbdce2d9b42658fa3ecb19c48261c/artifacts/weights/best.pt" ^
    --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.39 --name "yolo11-obb-X#mlflow-685cbdce2d9b42658fa3ecb19c48261c" --batch-size 32 --max-det 100 ^
    --project-name "wildai-validation" --save-txt --plots

call uv run tools\validate.py --splits "test" --is-detector --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
    --weights "runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt" ^
    --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.32 --name "yolo11-obb-s#mlflow-f5b7124be14c4c89b8edd26bcf7a9a76" --batch-size 32 --max-det 100 ^
    --project-name "wildai-validation" --save-txt --plots



@REM MMROTATE
call deactivate

call .venv-mmrotate\Scripts\activate.bat

call uv run tools\validate_mmrotate.py "runs-mmrotate\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_1x_dota_le90.py" "runs-mmrotate\oriented_rcnn_r50_fpn\best_mAP_epoch_13.pth" ^
        --eval "mAP" --work-dir "runs-mmrotate\val\oriented_rcnn_r50_fpn_1x_dota_le90" --fuse-conv-bn 

call uv run tools\validate_mmrotate.py "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn#empty_2.5#freeze_0.5\roi_trans_r50_fpn_1x_dota_ms_rr_le90.py" "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn#empty_2.5#freeze_0.5\best_mAP_epoch_2.pth" ^
        --eval "mAP" --work-dir "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn#empty_2.5#freeze_0.5" --fuse-conv-bn 

call uv run tools\validate_mmrotate.py "runs-mmrotate\redet_re50_fpn\redet_re50_fpn#empty_2.5#freeze_0.5\redet_re50_refpn_1x_dota_le90.py" "runs-mmrotate\redet_re50_fpn\redet_re50_fpn#empty_2.5#freeze_0.5\best_mAP_epoch_1.pth" ^
        --eval "mAP" --work-dir "runs-mmrotate\redet_re50_fpn\redet_re50_fpn#empty_2.5#freeze_0.5" --fuse-conv-bn



@REM @REM  PaddlePaddle Detection
call deactivate

call .venv-paddle\Scripts\activate

call uv run tools\validate_ppd.py -c "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone.yml" ^
        -o use_gpu=true weights="runs_ppd\ppyoloe_plus_sod_crn_s_80e_visdrone\best_model\model.pdparams" PPYOLOEHead.nms.score_threshold=0.4 EvalReader.batch_size=8 TestReader.batch_size=16 ^
        --classwise --amp --output_eval "runs_ppd\ppyoloe_plus_sod_crn_s_80e_visdrone"  --mode "test"


call uv run tools\validate_ppd.py -c "configs\ppd_configs\ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml" ^
        -o use_gpu=true weights="runs_ppd\ppyoloe_plus_sod_crn_l_largesize_80e_visdronetr_empty_ratio_0.33_freeze_0.5\best_model\model.pdparams" PPYOLOEHead.nms.score_threshold=0.4 EvalReader.batch_size=16 TestReader.batch_size=16 ^
        --classwise --amp --output_eval "runs_ppd\ppyoloe_plus_sod_crn_l_largesize_80e_visdronetr_empty_ratio_0.33_freeze_0.5"  --mode "test" 




