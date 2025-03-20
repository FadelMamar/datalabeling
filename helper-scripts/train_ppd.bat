
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call .venv-paddle\Scripts\activate.bat

@REM call wandb online

call uv run tools\train_paddlepaddle.py --epoch 30 --lr0 0.0005 --amp --output-dir "runs_ppd\ppyoloe_plus_sod_crn_s_80e_visdrone" --use-wandb --project-name "wildAI-detection"^
                                  --run-name "run-ppd" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone.yml" --do-eval True^
                                  --weights "base_models_weights\ppyoloe_plus_sod_crn_s_80e_visdrone.pdparams"  --tags "debug"

@REM call deactivate