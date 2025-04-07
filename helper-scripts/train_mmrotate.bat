
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call .venv-mmrotate\Scripts\activate.bat

call wandb online

@REM call uv run python -W once tools\train-mmrotate.py --epoch 7 --batch-size 16 --enable-val ^
@REM     --lr0 0.0005 --config "configs\mmrotate_configs\oriented_rcnn_r50_fpn_1x_dota_le90.py" ^
@REM     --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --project-name "wildAI-detection" --use-wandb ^
@REM     --weights "runs-mmrotate\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75\best_mAP_epoch_12.pth" ^
@REM     --run-name "oriented_rcnn_r50_fpn_identif" --output-dir runs-mmrotate\oriented_rcnn_r50_fpn ^
@REM     --tags "identification" --empty-ratio 2.5 --freeze-ratio 0.75

@REM call uv run python -W once tools\train-mmrotate.py --epoch 7 --batch-size 16 --enable-val ^
@REM     --lr0 0.0005 --config "configs\mmrotate_configs\redet_re50_refpn_1x_dota_le90.py" ^
@REM     --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --project-name "wildAI-detection" --use-wandb ^
@REM     --weights "runs-mmrotate\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75\best_mAP_epoch_13.pth" ^
@REM     --run-name "redet_re50_fpn_identif" --output-dir runs-mmrotate\redet_re50_fpn ^
@REM     --tags "identification" --empty-ratio 2.5 --freeze-ratio 0.75

@REM call uv run python -W once tools\train-mmrotate.py --epoch 7 --batch-size 16 --enable-val ^
@REM     --lr0 0.0005 --config "configs\mmrotate_configs\roi_trans_r50_fpn_1x_dota_ms_rr_le90.py" ^
@REM     --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --project-name "wildAI-detection" --use-wandb ^
@REM     --weights "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75\best_mAP_epoch_12.pth" ^
@REM     --run-name "roi_trans_r50_fpn_identif" --output-dir runs-mmrotate\roi_trans_r50_fpn ^
@REM     --tags "identification" --empty-ratio 2.5 --freeze-ratio 0.75

@REM call uv run python -W once tools\train-mmrotate.py --epoch 10 --batch-size 16 --enable-val ^
@REM     --lr0 0.001 --config "configs\mmrotate_configs\oriented_rcnn_r50_fpn_1x_dota_le90.py" ^
@REM     --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --project-name "wildAI-detection" --use-wandb ^
@REM     --weights "runs-mmrotate\oriented_rcnn_r50_fpn\best_mAP_epoch_13.pth" ^
@REM     --run-name "oriented_rcnn_r50_fpn" --output-dir runs-mmrotate\oriented_rcnn_r50_fpn ^
@REM     --tags "mmrotate" "detection" --empty-ratio 2.5 --freeze-ratio 0.5

@REM call uv run python -W once tools\train-mmrotate.py --epoch 10 --batch-size 16 --enable-val ^
@REM     --lr0 0.0005 --config "configs\mmrotate_configs\redet_re50_refpn_1x_dota_le90.py" ^
@REM     --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --project-name "wildAI-detection" --use-wandb ^
@REM     --weights "runs-mmrotate\redet_re50_fpn\latest.pth" ^
@REM     --run-name "redet_re50_fpn" --output-dir runs-mmrotate\redet_re50_fpn ^
@REM     --tags "mmrotate" "detection" --empty-ratio 2.5 --freeze-ratio 0.5

@REM call uv run python -W once tools\train-mmrotate.py --epoch 10 --batch-size 16 --enable-val ^
@REM     --lr0 0.0005 --config "configs\mmrotate_configs\roi_trans_r50_fpn_1x_dota_ms_rr_le90.py" ^
@REM     --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --project-name "wildAI-detection" --use-wandb ^
@REM     --weights "runs-mmrotate\roi_trans_r50_fpn\latest.pth" ^
@REM     --run-name "roi_trans_r50_fpn" --output-dir runs-mmrotate\roi_trans_r50_fpn ^
@REM     --tags "mmrotate" "detection" --empty-ratio 2.5 --freeze-ratio 0.5

call deactivate