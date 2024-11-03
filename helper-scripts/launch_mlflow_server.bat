::call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
::call conda activate label-backend

call cd ..

call mlflow server --backend-store-uri runs\mlflow 
