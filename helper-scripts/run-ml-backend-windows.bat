call .\activate_label-backend_env.bat

call conda env config vars set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\
call conda env config vars set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
call conda env config vars set LOCAL_FILES_DOCUMENT_ROOT=D:\

call conda deactivate
call conda activate label-backend

start .\launch_mlflow_server.bat


call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\my_ml_backend"


call label-studio-ml start ../my_ml_backend -p 9090

:: deactivate
:: call "C:\Users\Machine Learning\Desktop\workspace-wildAI\envs\label-studio\Scripts\deactivate.bat"
call conda deactivate