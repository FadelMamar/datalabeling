import streamlit as st
import pandas as pd
from typing import List
import requests

# Configuration
LABEL_STUDIO_URL = "http://localhost:8080"
TRAINING_API_URL = "http://training-service:8000"


def main():
    st.set_page_config(page_title="Labeling Workflow Manager", layout="wide")

    st.title("Labeling Workflow Management")

    # Sidebar for common controls
    with st.sidebar:
        st.header("API Configuration")
        label_studio_token = st.text_input("Label Studio Token", type="password")
        training_api_token = st.text_input("Training API Token", type="password")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        ["Upload Annotations", "Project Analytics", "Model Training"]
    )

    with tab1:
        st.header("Upload to Label Studio")
        with st.form("upload_annotations"):
            project_id = st.number_input("Project ID", min_value=1, step=1)
            model_alias = st.text_input("Model Alias")
            annotation_file = st.file_uploader("Annotation File (JSON)", type=["json"])

            if st.form_submit_button("Upload Annotations"):
                try:
                    response = upload_to_label_studio(
                        project_id, model_alias, annotation_file, label_studio_token
                    )
                    st.success(f"Upload successful! Response: {response.status_code}")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    with tab2:
        st.header("Project Analytics")
        with st.form("project_stats"):
            stats_project_id = st.number_input(
                "Analytics Project ID", min_value=1, step=1
            )

            if st.form_submit_button("Show Statistics"):
                try:
                    stats_df = get_project_statistics(
                        stats_project_id, label_studio_token
                    )
                    st.dataframe(stats_df, use_container_width=True)

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Total Annotations", stats_df["total_annotations"].iloc[0]
                    )
                    col2.metric("Completed Tasks", stats_df["completed_tasks"].iloc[0])
                    col3.metric(
                        "Avg Quality", f"{stats_df['avg_quality'].iloc[0]:.2f}%"
                    )

                except Exception as e:
                    st.error(f"Failed to fetch statistics: {str(e)}")

    with tab3:
        st.header("Train Object Detector")
        with st.form("model_training"):
            training_projects = st.text_input("Project IDs (comma-separated)")
            epochs = st.slider("Training Epochs", 1, 100, 10)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64])

            if st.form_submit_button("Start Training"):
                try:
                    project_ids = [
                        int(pid.strip()) for pid in training_projects.split(",")
                    ]
                    response = start_training(
                        project_ids, epochs, batch_size, training_api_token
                    )
                    st.success(
                        f"Training initiated! Job ID: {response.json()['job_id']}"
                    )
                    st.json(response.json())
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")


# Mock API client functions (implement according to your API specs)
def upload_to_label_studio(
    project_id: int, model_alias: str, annotation_file, token: str
):
    """Mock function for Label Studio upload"""
    headers = {"Authorization": f"Token {token}"}
    files = {"file": annotation_file.getvalue()}
    return requests.post(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import",
        headers=headers,
        files=files,
        params={"model_alias": model_alias},
    )


def get_project_statistics(project_id: int, token: str) -> pd.DataFrame:
    """Mock function for statistics retrieval"""
    headers = {"Authorization": f"Token {token}"}
    response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/stats", headers=headers
    )
    return pd.DataFrame([response.json()])


def start_training(project_ids: List[int], epochs: int, batch_size: int, token: str):
    """Mock function for training initialization"""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"project_ids": project_ids, "epochs": epochs, "batch_size": batch_size}
    return requests.post(f"{TRAINING_API_URL}/train", headers=headers, json=payload)


if __name__ == "__main__":
    main()
