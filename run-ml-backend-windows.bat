
call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

:: call set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=True
:: call set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="D:\"

label-studio-ml start ./my_ml_backend -p 9090

:: deactivate
call "C:\Users\Machine Learning\Desktop\workspace-wildAI\envs\label-studio\Scripts\deactivate.bat"