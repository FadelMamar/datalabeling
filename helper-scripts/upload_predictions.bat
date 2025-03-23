call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

@REM --project-id
@REM --alias
@REM --confidence-threshold
call uv run upload_predictions.py --project-id 0^
                                  --alias "version14" "version11"^
                                  --confidence-threshold 0.15

call deactivate