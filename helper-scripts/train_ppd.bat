
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call .venv-paddle\Scripts\activate

call wandb online

@REM Identification

call uv run tools\train_paddlepaddle.py --epoch 7 --lr0 0.0005 --amp --tr-batchsize 16  --use-wandb --project-name "wildAI-detection" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
                                  --do-eval True --eval-interval 2 --val-empty-ratio 0.4 --tr-empty-ratio 2.5 --freeze-ratio 0.75 ^
                                  --run-name "ppyoloe_plus_sod_crn_s_80e_visdrone_identif" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone_identification.yml" ^
                                  --weights "..."  --tags "identification"

call uv run tools\train_paddlepaddle.py --epoch 7 --lr0 0.0005 --amp --tr-batchsize 16  --use-wandb --project-name "wildAI-detection" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
                                  --do-eval True --eval-interval 2 --val-empty-ratio 0.4 --tr-empty-ratio 2.5 --freeze-ratio 0.75 ^
                                  --run-name "ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_identif" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_identification.yml" ^
                                  --weights ".."  --tags "identification"


@REM Detection

call uv run tools\train_paddlepaddle.py --epoch 7 --lr0 0.0005 --amp --tr-batchsize 16  --use-wandb --project-name "wildAI-detection" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
                                  --do-eval True --eval-interval 2 --val-empty-ratio 0.4 --tr-empty-ratio 2.5 --freeze-ratio 0.75 ^
                                  --run-name "ppyoloe_plus_sod_crn_s_80e_visdrone" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone.yml" ^
                                  --weights "..."  --tags "detection"

call uv run tools\train_paddlepaddle.py --epoch 7 --lr0 0.0005 --amp --tr-batchsize 16  --use-wandb --project-name "wildAI-detection" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
                                  --do-eval True --eval-interval 2 --val-empty-ratio 0.4 --tr-empty-ratio 2.5 --freeze-ratio 0.75 ^
                                  --run-name "ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_identif" --config "configs\ppd_configs\ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml" ^
                                  --weights "..."  --tags "detection"




@REM call deactivate
