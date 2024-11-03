call .\activate_label-backend_env.bat

call cd ..\tools

call wandb online

call python sweeps.py

call conda deactivate