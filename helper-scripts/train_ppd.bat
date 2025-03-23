
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call D:\PaddleDetection\.venv\Scripts\activate

call wandb offline

call uv run tools\train_paddlepaddle.py --epoch 30 --lr0 0.001 --amp --output-dir "runs_ppd\ppyoloe_plus_sod_crn_s_80e_visdrone" --use-wandb --project-name "wildAI-detection"^
                                  --run-name "run-ppd" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone.yml" --do-eval True^
                                  --weights "base_models_weights\ppyoloe_plus_sod_crn_s_80e_visdrone.pdparams"  --tags "debug" --device "cpu"^
                                  --data-config "configs\yolo_configs\data_config.yaml" --tr-empty-ratio 0.1 --val-empty-ratio 0.3 --print-flops

@REM call deactivate