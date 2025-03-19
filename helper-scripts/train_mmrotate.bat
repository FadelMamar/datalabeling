
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call .venv-mmrotate\Scripts\activate.bat

@REM call wandb online

call uv run tools\train-mmrotate.py --epoch 30 ^
    --lr0 0.001 --config notebooks\oriented_rcnn_r50_fpn_1x_dota_le90.py ^
    --weights notebooks\oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth ^
    --run-name "oriented_rcnn_r50_fpn" --output-dir runs-mmrotate\oriented_rcnn_r50_fpn ^
    --tags "mmrotate" "detection" --empty-ratio 0.1 --data-config data\data_config.yaml

@REM call deactivate