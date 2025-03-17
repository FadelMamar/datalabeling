import yaml
from ..arguments import Arguments
import os
import traceback
from pathlib import Path
import pandas as pd

__all__ = ["remove_label_cache",
           "sample_pos_neg",
           "get_data_cfg_paths_for_cl",
           "get_data_cfg_paths_for_HN",
          ]

def remove_label_cache(data_config_yaml: str):
    # Remove labels.cache
    with open(data_config_yaml, "r") as file:
        yolo_config = yaml.load(file, Loader=yaml.FullLoader)
    root = yolo_config["path"]
    for split in ["train", "val", "test"]:
        try:
            for p in yolo_config[split]:
                path = os.path.join(root, p, "../labels.cache")
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Removing: {os.path.join(root, p, '../labels.cache')}")
                # else:
                #     print(path, "does not exist.")
        except Exception:
            # print(e)
            traceback.print_exc()


def sample_pos_neg(images_paths: list, ratio: float, seed: int = 41):
    """_summary_

    Args:
        images_paths (list): images paths
        ratio (float): ratio defined as num_empty/num_non_empty
        seed (int, optional): random seed. Defaults to 41.

    Returns:
        list: selected paths to images
    """

    # build dataframe
    is_empty = [
        1 - Path(str(p).replace("images", "labels")).with_suffix(".txt").exists()
        for p in images_paths
    ]
    data = pd.DataFrame.from_dict(
        {"image_paths": images_paths, "is_empty": is_empty}, orient="columns"
    )
    # get empty and non empty
    num_empty = (data["is_empty"] == 1).sum()
    num_non_empty = len(data) - num_empty
    if num_empty == 0:
        print("contains only positive samples")
    num_sampled_empty = min(int(num_non_empty * ratio), num_empty)
    sampled_empty = data.loc[data["is_empty"] == 1].sample(
        n=num_sampled_empty, random_state=seed
    )
    # concatenate
    sampled_data = pd.concat([sampled_empty, data.loc[data["is_empty"] == 0]])

    print(f"Sampling: pos={num_non_empty} & neg={num_sampled_empty}", end="\n")

    return sampled_data["image_paths"].to_list()


def get_data_cfg_paths_for_cl(
    ratio: float,
    data_config_yaml: str,
    cl_save_dir: str,
    seed: int = 41,
    split: str = "train",
    pattern_glob: str = "*",
):
    """_summary_

    Args:
        ratio (float): _description_
        data_config_yaml (str): _description_
        cl_save_dir (str): _description_
        seed (int, optional): _description_. Defaults to 41.
        split (str, optional): _description_. Defaults to 'train'.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    with open(data_config_yaml, "r") as file:
        yolo_config = yaml.load(file, Loader=yaml.FullLoader)

    root = yolo_config["path"]
    train_dirs_images = [os.path.join(root, p) for p in yolo_config[split]]

    # sample positive and negative images
    sampled_imgs_paths = []
    for dir_images in train_dirs_images:
        print(f"Sampling positive and negative samples from {dir_images}")
        paths = sample_pos_neg(
            images_paths=list(Path(dir_images).glob(pattern_glob)),
            ratio=ratio,
            seed=seed,
        )
        sampled_imgs_paths = sampled_imgs_paths + paths

    # save selected images in txt file
    save_path_samples = os.path.join(
        cl_save_dir, f"{split}_ratio_{ratio}-seed_{seed}.txt"
    )
    pd.Series(sampled_imgs_paths).to_csv(save_path_samples, index=False, header=False)
    print(f"Saving {len(sampled_imgs_paths)} sampled images.")
    # save config
    if split == "train":
        cfg_dict = {
            "path": root,
            "names": yolo_config["names"],
            "train": os.path.relpath(save_path_samples, start=root),
            "val": yolo_config["val"],
            "nc": yolo_config["nc"],
        }
    elif split == "val":
        cfg_dict = {
            "path": root,
            "names": yolo_config["names"],
            "val": os.path.relpath(save_path_samples, start=root),
            "train": yolo_config["val"],
            "nc": yolo_config["nc"],
        }
    else:
        raise NotImplementedError
    save_path_cfg = Path(save_path_samples).with_suffix(".yaml")
    with open(save_path_cfg, "w") as file:
        yaml.dump(cfg_dict, file)

    print(
        f"Saving samples at: {save_path_samples} and data_cfg at {save_path_cfg}",
        end="\n\n",
    )

    return str(save_path_cfg)


def get_data_cfg_paths_for_HN(args: Arguments, data_config_yaml: str):
    """_summary_

    Args:
        args (Arguments): _description_
        data_config_yaml (str): _description_

    Returns:
        _type_: _description_
    """

    from ..annotator import Detector
    from ..dataset.sampling import (
        get_preds_targets,
        compute_detector_performance,
        get_uncertainty,
    )

    split = "train"
    pred_results_dir = args.hn_save_dir
    save_path_samples = os.path.join(args.hn_save_dir, "hard_samples.txt")
    data_config_root = "D:\\"
    save_data_config_yaml = os.path.join(args.hn_save_dir, "hard_samples.yaml")

    # Define detector
    model = Detector(
        path_to_weights=args.path_weights,
        confidence_threshold=args.hn_confidence_threshold,
        overlap_ratio=args.hn_overlap_ratio,
        tilesize=args.hn_tilesize,
        imgsz=args.hn_imgsz,
        use_sliding_window=args.hn_use_sliding_window,
        device=args.device,
        is_yolo_obb=args.hn_is_yolo_obb,
    )

    # data config yaml
    with open(data_config_yaml, "r") as file:
        yolo_config = yaml.load(file, Loader=yaml.FullLoader)
    # get images_paths
    images_paths = os.path.join(yolo_config["path"], yolo_config[split])
    images_paths = pd.read_csv(images_paths, header=None, names=["paths"])[
        "paths"
    ].to_list()
    # get predictions & targets
    df_results, df_labels, col_names = get_preds_targets(
        images_dirs=None,
        images_paths=images_paths,
        pred_results_dir=pred_results_dir,
        detector=model,
        load_results=args.hn_load_results,
        save_tag="hn-sampling",
    )
    # compute performance & uncertainty of model
    df_results_per_img = compute_detector_performance(df_results, df_labels, col_names)
    df_results_per_img = get_uncertainty(
        df_results_per_img=df_results_per_img, mode=args.hn_uncertainty_method
    )

    # save hard samples. Those with low mAP and high or low confidence score
    score_col = "max_scores"
    mask_low_map = (df_results_per_img["map50"] < args.hn_map_thrs) * (
        df_results_per_img["map75"] < args.hn_map_thrs
    )
    mask_high_scores = df_results_per_img[score_col] > args.hn_score_thrs
    mask_low_scores = df_results_per_img[score_col] < (1 - args.hn_score_thrs)
    mask_selected = (
        mask_low_map * mask_high_scores
        + mask_low_map * mask_low_scores
        + (df_results_per_img["uncertainty"] > args.hn_uncertainty_thrs)
    )
    df_hard_negatives = df_results_per_img.loc[mask_selected]

    # save image paths in data_config yaml
    df_hard_negatives["image_paths"].to_csv(
        save_path_samples, index=False, header=False
    )
    cfg_dict = {
        "path": data_config_root,
        "names": yolo_config["names"],
        "train": os.path.relpath(save_path_samples, start=data_config_root),
        "val": [
            os.path.relpath(
                os.path.join(yolo_config["path"], p), start=data_config_root
            )
            for p in yolo_config["val"]
        ],
        "nc": yolo_config["nc"],
    }
    with open(save_data_config_yaml, "w") as file:
        yaml.dump(cfg_dict, file)

    return str(save_data_config_yaml)
