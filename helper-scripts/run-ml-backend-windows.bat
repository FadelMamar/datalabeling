@REM call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
@REM call conda activate label-studio

call deactivate

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call helper-scripts\activate_label-backend_env.bat

@REM call conda env config vars set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\
@REM call conda env config vars set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
@REM call conda env config vars set LOCAL_FILES_DOCUMENT_ROOT=D:\
call set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\
call set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
call set LOCAL_FILES_DOCUMENT_ROOT=D:\

@REM call conda deactivate
@REM call conda activate label-backend

start .\launch_mlflow_server.bat


call label-studio-ml start ./my_ml_backend -p 9090

:: deactivate
:: call "C:\Users\Machine Learning\Desktop\workspace-wildAI\envs\label-studio\Scripts\deactivate.bat"
call deactivate
