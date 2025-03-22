
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call .venv-paddle\Scripts\activate.bat

call wandb online

call uv run tools\train_paddlepaddle.py --epoch 30 --lr0 0.0005 --amp  --use-wandb --project-name "wildAI-detection"^
                                  --do-eval True --eval-interval 2 --output-dir "runs_ppd\ppyoloe_plus_sod_crn_s_80e_visdrone" ^
                                  --run-name "ppyoloe_plus_sod_crn_s_80e_visdrone" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone.yml" ^
                                  --weights "base_models_weights\ppyoloe_plus_sod_crn_s_80e_visdrone.pdparams"  --tags "debug"

@REM call deactivate