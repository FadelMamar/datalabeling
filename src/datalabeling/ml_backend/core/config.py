# src/ml_backend/core/config.py
from pydantic import BaseSettings


class Settings(BaseSettings):
    model_name: str = "logistic_regression"
    data_path: str = "data/train.csv"
    label_column: str = "target"
    api_prefix: str = "/api"


# settings = Settings()
