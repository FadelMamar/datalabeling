@REM call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
@REM call conda activate label-studio
call "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\label-studio\Scripts\activate"

@REM call conda env config vars set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\
@REM call conda env config vars set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
@REM call conda env config vars set LOCAL_FILES_DOCUMENT_ROOT=D:\
call set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\
call set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
call set LOCAL_FILES_DOCUMENT_ROOT=D:\

@REM call conda deactivate
@REM call conda activate label-studio


call label-studio start -p 8080

:: deactivate.. Should be updated to match correct location of deactivate.bat
:: call "C:\Users\Machine Learning\Desktop\workspace-wildAI\envs\label-studio\Scripts\deactivate.bat"
@REM @REM call conda deactivate
call deactivate