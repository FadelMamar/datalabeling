import streamlit as st
import pandas as pd
from typing import List
import requests
from dotenv import load_dotenv
from pathlib import Path
import os
import traceback
from datalabeling.ml.interface import Annotator
from datalabeling.common.io import load_yaml
from datalabeling.common.annotation_utils import GPSUtils, ImageProcessor
import logging
from label_studio_sdk.client import LabelStudio
from itertools import chain

DOT_ENV = Path(__file__) / "../.env"
load_dotenv(DOT_ENV)
LABEL_STUDIO_URL = os.environ["LABEL_STUDIO_URL"]
LABEL_STUDIO_API_KEY = os.environ["LABEL_STUDIO_API_KEY"]
LABEL_STUDIO_CLIENT = LabelStudio(
    base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY
)
TRAINING_API_URL = ...
TRAINING_API_KEY = ...
# LOCAL_FILES_DOCUMENT_ROOT = os.environ['LOCAL_FILES_DOCUMENT_ROOT']


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_func):
        super().__init__()
        self.widget = widget_func  # e.g. st.empty().code

    def emit(self, record):
        msg = self.format(record)
        self.widget(msg)


def main():
    st.set_page_config(page_title="Labeling Workflow Manager", layout="wide")

    st.title("Labeling Workflow Management")

    # Sidebar for common controls
    # with st.sidebar:
    #     st.header("API Configuration")
    # label_studio_token = st.text_input("Label Studio Token", type="password")
    # training_api_token = st.text_input("Training API Token", type="password")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Upload Annotations",
            "Project Analytics",
            "Model Training",
            "GPS",
            "Inference",
        ]
    )

    with tab1:
        st.header("Upload to Label Studio")
        with st.form("upload_annotations"):
            project_id = st.number_input("Project ID", min_value=0, step=1)
            model_alias = st.text_input("Model Alias").strip()
            detector_name = st.text_input("Detector name", value="obb-detector").strip()
            confidence_threshold = st.text_input(
                "Confidence threshold", value=0.2
            ).strip()
            path_to_weights = st.text_input(
                "Path to model weights",
                # value=r"D:\datalabeling\base_models_weights\best.pt",
            ).strip()
            tile_size = st.number_input("Tile size", min_value=640, step=1)
            top_n = st.number_input("Top n", min_value=0, step=1)
            use_sliding_window = True
            # annotation_file = st.file_uploader("Annotation File (JSON)", type=["json"])

            annotator_kwargs = {
                "path_to_weights": path_to_weights,
                "mlflow_model_alias": model_alias,
                "mlflow_model_name": detector_name,
                "tilesize": tile_size,
                "overlapratio": 0.1,
                "use_sliding_window": True,
                "is_yolo_obb": detector_name == "obb-detector",
                "confidence_threshold": 0.1,
                "tag_to_append": "",
            }

            log_widget = st.empty().code
            handler = StreamlitLogHandler(log_widget)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            if st.form_submit_button("Upload Annotations"):
                try:
                    with st.spinner("Uploading annotations...", show_time=True):
                        upload_to_label_studio(
                            project_id=project_id, top_n=top_n, **annotator_kwargs
                        )
                        st.success("Done!")

                except Exception as e:
                    traceback.print_exc()
                    # st.error(f"Upload failed: {str(e)}")

    with tab2:
        st.header("Project Analytics")
        with st.form("project_stats"):
            stats_project_id = st.number_input(
                "Analytics Project ID", min_value=0, step=1
            )

            log_widget = st.empty().code
            handler = StreamlitLogHandler(log_widget)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            if st.form_submit_button("Show Statistics"):
                try:
                    with st.spinner("Computing statistics...", show_time=True):
                        instances_count, images_count = get_project_statistics(
                            stats_project_id
                        )
                    st.dataframe(instances_count, use_container_width=False)
                    st.dataframe(images_count, use_container_width=False)

                    # Display metrics
                    # col1, col2, col3 = st.columns(len(images_count.columns))
                    # col1.metric(
                    #     "Total Annotations", stats_df["total_annotations"].iloc[0]
                    # )
                    # col2.metric("Completed Tasks", stats_df["completed_tasks"].iloc[0])
                    # col3.metric(
                    #     "Avg Quality", f"{stats_df['avg_quality'].iloc[0]:.2f}%"
                    # )

                except Exception as e:
                    st.error(f"Failed to fetch statistics: {str(e)}")

        with st.form("train_val_test_stats"):
            path_to_yaml = st.text_input(
                "Path to data.yaml file",
                value=r"D:\datalabeling\configs\yolo_configs\data_config.yaml",
            ).strip()
            split = st.text_input(
                "Split to select", value="train", help="train val or test"
            ).strip()

            if st.form_submit_button("Show Statistics"):
                with st.spinner("Computing statistics...", show_time=True):
                    stats = visualize_splits_distribution(
                        data_yaml_path=path_to_yaml, split=split
                    )

                st.bar_chart(
                    stats["instances_count"],
                    x="class",
                    y="count",
                    x_label="Species",
                    y_label="Instance count",
                )
                st.bar_chart(
                    stats["images_count"],
                    x="class",
                    y="image",
                    x_label="Species",
                    y_label="Images count",
                )

    with tab3:
        st.header("Train Object Detector")
        with st.form("model_training"):
            training_projects = st.text_input("Project IDs (comma-separated)")
            epochs = st.slider("Training Epochs", 1, 100, 10)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64])

            if st.form_submit_button("Start Training"):
                raise NotImplementedError

    with tab4:
        st.header("GPS")
        with st.form("gps_coords"):
            image_dir = st.text_input(
                "Path to images directory (without quotes)"
            ).strip()

            if st.form_submit_button("Get coordinates"):
                gps_coords = get_gps_coords(image_paths=None, image_dir=image_dir)
                st.dataframe(gps_coords, use_container_width=False)

    with tab5:
        st.header("Inference")

        with st.form("inference"):
            model_alias = st.text_input("Model Alias", value="version18").strip()
            model_name = st.text_input("Model name", value="obb-detector").strip()
            confidence_threshold = st.text_input(
                "Confidence threshold", value=0.15
            ).strip()
            image_dir = st.text_input(
                "Path to images directory (without quotes)"
            ).strip()
            save_path = st.text_input("Save path (without quotes)").strip()

            if st.form_submit_button("Get predictions"):
                run_inference(
                    image_dir=image_dir,
                    alias=model_alias,
                    save_path=save_path,
                    image_paths=None,
                    confidence_threshold=confidence_threshold,
                    dotenv_path=DOT_ENV,
                    name=model_name,
                    exts=[
                        "*.jpg",
                        "*.jpeg",
                        "*.png",
                    ],
                )


def run_inference(
    image_dir: str,
    alias: str,
    save_path: str,
    image_paths: list[None] = None,
    confidence_threshold: float = 0.15,
    dotenv_path: str = "../.env",
    name: str = "obb-detector",
    exts: list[str] = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ],
) -> None:
    handler = Annotator(
        mlflow_model_alias=alias,
        mlflow_model_name=name,
        confidence_threshold=confidence_threshold,
        is_yolo_obb=name.strip() == "obb-detector",
        dotenv_path=dotenv_path,
    )

    exts = [e.lower() for e in exts] + [e.capitalize() for e in exts]

    if image_paths is None:
        image_paths = chain.from_iterable([Path(image_dir).glob(ext) for ext in exts])

    results = handler.predict_directory(
        path_to_dir=None,
        images_paths=image_paths,
        return_gps=True,
        return_coco=False,
        as_dataframe=True,
        save_path=None,
    )

    results[["Latitude", "Longitude", "Elevation"]].to_csv(save_path, index=False)


def get_gps_coords(
    image_dir: str,
    image_paths: list[str] = None,
    exts: list[str] = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ],
):
    exts = [e.lower() for e in exts] + [e.capitalize() for e in exts]

    if image_paths is None:
        image_paths = chain.from_iterable([Path(image_dir).glob(ext) for ext in exts])

    gps_coords = [
        GPSUtils.get_gps_coord(file_name=path, return_as_decimal=True)[0]
        for path in image_paths
    ]

    gps_coords = pd.DataFrame(
        data=gps_coords, columns=["Latitude", "Longitude", "Elevation"]
    )

    return gps_coords


# Mock API client functions (implement according to your API specs)
def upload_to_label_studio(project_id: int, top_n: int = 0, **annotator_kwargs):
    handler = Annotator(dotenv_path=str(DOT_ENV), **annotator_kwargs)
    handler.upload_predictions(project_id=project_id, top_n=top_n)


def get_project_statistics(
    project_id: int, annotator_id=0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    instances_count, images_count = Annotator.get_project_stats(
        LABEL_STUDIO_CLIENT, project_id=project_id, annotator_id=annotator_id
    )

    instances_count.rename(
        columns={col: col + "_num_instances" for col in instances_count.columns},
        inplace=True,
    )
    images_count.rename(
        columns={col: col + "_num_images" for col in images_count.columns}, inplace=True
    )

    return instances_count, images_count


def visualize_splits_distribution(
    data_yaml_path: str,
    split="train",
):
    from tqdm import tqdm

    logger = logging.getLogger(__file__)

    # load yaml
    yolo_config = load_yaml(data_yaml_path)

    label_map = yolo_config["names"]

    path_dataset = [os.path.join(yolo_config["path"], p) for p in yolo_config[split]]

    # iter_labels = path_dataset.glob("*.txt")

    iter_labels = chain.from_iterable(
        [Path(p.replace("images", "labels")).glob("*.txt") for p in path_dataset]
    )

    # total_number_images = len(list(iter_labels))
    # path_dataset = path_dataset.replace('images','labels')
    # total_number_of_positive_images = len(list(Path(path_dataset).glob('*')))

    labels = list()
    for txtfile in tqdm(iter_labels, desc="Reading labels from data.yaml"):
        df = pd.read_csv(txtfile, sep=" ", header=None)
        df["class"] = df.iloc[:, 0].astype(int)
        df["image"] = txtfile.stem
        labels.append(df)

    df = pd.concat(labels, axis=0).reset_index(drop=True)
    df["class"] = df["class"].map(label_map)

    images_count = df.groupby("class")["image"].count().reset_index()
    instances_count = df["class"].value_counts().reset_index()

    #
    # stats = dict(num_negative=total_number_images-total_number_of_positive_images,
    #              num_positive=total_number_of_positive_images
    #              )
    stats = dict()
    stats.update({"instances_count": instances_count, "images_count": images_count})

    return stats


def start_training(project_ids: List[int], epochs: int, batch_size: int, token: str):
    """Mock function for training initialization"""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"project_ids": project_ids, "epochs": epochs, "batch_size": batch_size}
    return requests.post(f"{TRAINING_API_URL}/train", headers=headers, json=payload)


if __name__ == "__main__":
    main()
