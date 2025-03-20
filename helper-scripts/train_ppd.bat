
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call .venv-paddle\Scripts\activate.bat

@REM call wandb online

call uv run tools\paddlepaddle.py --epoch 30 --lr0 0.0005 --amp --output-dir "runs_ppd" --use-wandb --project-name "wildAI-detection"^
                                  --run-name "run-ppd" --tags "debug"

@REM call deactivate