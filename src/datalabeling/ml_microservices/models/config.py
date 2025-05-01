from pydantic import BaseSettings


class PredictionConfig(BaseSettings):
    slice_width: int = 640
    slice_height: int = 640
    overlap_ratio: float = 0.2
    min_area_ratio: float = 0.1
