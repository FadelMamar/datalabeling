from dotenv import load_dotenv
from datalabeling.annotator import Annotator
from typing import Sequence
import fire


def main(project_id: int, aliases: Sequence[str], confidence_threshold: float = 0.15):
    DOT_ENV_PATH = "../.env"
    load_dotenv(DOT_ENV_PATH)

    # aliases = ["version11",]
    # project_id = 40 # insert correct project_id by loooking at the url
    for alias in aliases:
        name = "obb-detector"  # detector, "obb-detector"
        handler = Annotator(
            mlflow_model_alias=alias,
            mlflow_model_name=name,
            confidence_threshold=confidence_threshold,
            is_yolo_obb=name.strip() == "obb-detector",
            dotenv_path=DOT_ENV_PATH,
        )
        handler.upload_predictions(project_id=project_id)


if __name__ == "__main__":
    fire.Fire(main)
