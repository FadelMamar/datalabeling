from dotenv import load_dotenv
import yaml
import os
from dataclasses import dataclass
from datargs import parse
from pathlib import Path

load_dotenv("../.env")
DATA_DIR = "D:\PhD\Data per camp\DetectionDataset\hard_samples"


@dataclass
class Args:
    path_to_weights: str

    confidence_threshold: float = 0.1
    overlap_ratio: float = 0.1
    tilesize: int = 1280
    use_sliding_window: bool = False
    device: str = "cpu"
    is_yolo_obb: bool = False

    data_config_yaml: str = r"..\data\dataset_labeler.yaml"
    split: str = "train"

    pred_results_dir: str = DATA_DIR
    load_results: bool = False

    map_thrs: float = 0.3
    score_thrs: float = 0.7
    save_path_samples: str = os.path.join(DATA_DIR, "hard_samples.txt")
    data_config_root: str = "D:\\"
    save_data_config_yaml: str = os.path.join(DATA_DIR, "hard_samples.yaml")
    hn_uncertainty_thrs: float = 4
    hn_uncertainty_method: str = "entropy"


if __name__ == "__main__":
    from datalabeling.annotator import Detector
    from datalabeling.dataset.sampling import (
        get_preds_targets,
        select_hard_samples,
        compute_detector_performance,
    )

    args = parse(Args)

    # creating paths if do not exist
    for p in [args.save_path_samples, args.save_data_config_yaml]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pred_results_dir).mkdir(parents=True, exist_ok=True)

    # Define detector
    model = Detector(
        path_to_weights=args.path_to_weights,
        confidence_threshold=args.confidence_threshold,
        overlap_ratio=args.overlap_ratio,
        tilesize=args.tilesize,
        use_sliding_window=args.use_sliding_window,
        device=args.device,
        is_yolo_obb=args.is_yolo_obb,
    )

    # load groundtruth
    with open(args.data_config_yaml, "r") as file:
        yolo_config = yaml.load(file, Loader=yaml.FullLoader)

    images_path = [
        os.path.join(yolo_config["path"], yolo_config[args.split][i])
        for i in range(len(yolo_config[args.split]))
    ]
    labels_path = [p.replace("images", "labels") for p in images_path]

    df_results, df_labels, col_names = get_preds_targets(
        images_dirs=images_path,
        pred_results_dir=args.pred_results_dir,
        detector=model,
        load_results=args.load_results,
    )

    df_results_per_img = compute_detector_performance(df_results, df_labels, col_names)

    # save hard samples
    df_hard_negatives = select_hard_samples(
        df_results_per_img=df_results_per_img,
        map_thrs=args.map_thrs,
        score_thrs=args.score_thrs,
        save_path_samples=args.save_path_samples,
        root=args.data_config_root,
        save_data_yaml=args.save_data_config_yaml,
        uncertainty_method=args.hn_uncertainty_method,
        uncertainty_thrs=args.hn_uncertainty_thrs,
    )
