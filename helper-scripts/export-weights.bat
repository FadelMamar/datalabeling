call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

call uv run tools\cli.py --export-format "engine" --width 1280 --height 1280 --export-batch-size 1^
             --export-model-weights ""^
             --half --dynamic

call deactivate