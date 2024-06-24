call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\tools"

call wandb online

call python cli.py --start-training --batchsize 32  --epochs 50 --copy-paste 0.0


call conda deactivate