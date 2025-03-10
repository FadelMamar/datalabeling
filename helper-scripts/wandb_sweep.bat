call .\activate_label-backend_env.bat

call cd ..\tools

call wandb online

call uv run sweeps.py

call deactivate