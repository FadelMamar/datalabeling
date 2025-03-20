call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call helper-scripts\activate_label-backend_env.bat

call mlflow server --backend-store-uri runs\mlflow 
