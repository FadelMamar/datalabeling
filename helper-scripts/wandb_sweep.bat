call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat


call cd ..\tools

call wandb online

call uv run sweeps.py

call deactivate