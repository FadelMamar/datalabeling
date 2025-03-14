# imports
from pathlib import Path
import numpy as np
import io
import json
import yaml
from typing import List, Dict
from label_studio_ml.utils import get_local_path
from sahi.utils.file import load_json
from sahi.slicing import slice_coco
from skimage.io import imread, imsave
import shutil
import math
import pandas as pd
from ..arguments import Dataprepconfigs
import os
from tqdm import tqdm
from label_studio_converter import Converter
from label_studio_sdk import Client
import traceback
import tempfile


def load_ls_annotations(input_dir: str) -> tuple[list, list]:
    """Loads label studio annotations

    Args:
        input_dir (str, optional): directory with label studio ``json-min`` annotations.

    Returns:
        tuple[list,list]: loaded_annotations,path_to_json_file
    """

    ls_annotations = []
    paths = list(Path(input_dir).glob("*.json"))

    for json_file in paths:
        with open(json_file, "r") as f:
            annotation = json.load(fp=f)
            ls_annotations.append(annotation)

    return ls_annotations, paths


def convert_lsjsonmin_to_df(json_data: List[Dict]) -> pd.DataFrame:
    """Converts Label studio (LS) annotations from json to DataFrames

    Args:
        json_data (List[Dict]): list of LS ``json-min`` annotations

    Returns:
        pd.DataFrame: converted annotations
    """

    # Reference for label studio output schema  https://labelstud.io/tags/rectanglelabels

    # placeholders
    image_paths = list()
    xs = list()
    ys = list()
    widths = list()
    heights = list()
    labels = list()
    image_widths = list()
    image_heights = list()

    for data in json_data:
        # load data
        if "label" in data.keys():
            image_path = get_local_path(url=data["image"])
            for rectangle in data["label"]:
                x = math.ceil(rectangle["x"] * rectangle["original_width"] / 100)
                y = math.ceil(rectangle["y"] * rectangle["original_height"] / 100)
                w = math.ceil(rectangle["width"] * rectangle["original_width"] / 100)
                h = math.ceil(rectangle["height"] * rectangle["original_height"] / 100)
                label = rectangle["rectanglelabels"][0]
                # update placeholders
                image_paths.append(image_path)
                image_heights.append(data["label"][0]["original_height"])
                image_widths.append(data["label"][0]["original_width"])
                xs.append(x)
                ys.append(y)
                widths.append(w)
                heights.append(h)
                labels.append(label)

    # csv data
    csv_data = {
        "images": image_paths,
        "x": xs,
        "y": ys,
        "width": widths,
        "height": heights,
        "labels": labels,
        "image_height": image_heights,
        "image_width": image_widths,
    }
    df = pd.DataFrame.from_dict(data=csv_data, orient="columns")
    df["x_min"] = df["x"].copy()
    df["x_max"] = df["x"] + df["width"].apply(int)
    df["y_min"] = df["y"].copy()
    df["y_max"] = df["y"] + df["height"].apply(int)
    df["x"] = df["x_min"] + (df["width"] * 0.5).apply(int)
    df["y"] = df["y_min"] + (df["height"] * 0.5).apply(int)

    return df


def convert_lsjsonmin_annotations_to_csv(
    input_dir, csv_dir_path, rewrite_existing=True
) -> None:
    """Converts LS json-min annotations to csv and saves to the same directory

    Args:
        rewrite_existing (bool, optional): states with the csv should be saved if it already exists. Defaults to True.
    """
    # load all available annotations
    annotations, paths = load_ls_annotations(input_dir=input_dir)

    # parse and save annotations
    for annotation, path in tqdm(
        zip(annotations, paths), desc="Converting jsons to csv"
    ):
        save_path = os.path.join(csv_dir_path, path.name.replace(".json", ".csv"))
        if Path(save_path).exists():
            if rewrite_existing:
                df = convert_lsjsonmin_to_df(json_data=annotation)
                df.to_csv(save_path, index=False)
        else:
            df = convert_lsjsonmin_to_df(json_data=annotation)
            df.to_csv(save_path, index=False)


def convert_json_to_coco(
    input_file: str,
    ls_xml_config: str,
    out_file_name: str = None,
    parsed_config: dict = None,
) -> Dict:
    """Converts LS json annotations to coco format

    Args:
        input_file (str): path to LS json annotation file
        out_file_name (str, optional): if not None, it will save the converted annotations. Defaults to None.

    Returns:
        dict: annotations in coco format
    """
    # load converter
    config_str = None
    if parsed_config is not None:
        assert ls_xml_config is None, "It has to be None"
        config_str = parsed_config
    elif ls_xml_config is not None:
        with io.open(ls_xml_config) as f:
            config_str = f.read()
    else:
        raise NotImplementedError("")
    handler = Converter(config=config_str, project_dir=None, download_resources=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        handler.convert_to_coco(
            input_data=input_file,
            output_dir=tmp_dir,
            output_image_dir=os.path.join(tmp_dir, "images"),
            is_dir=False,
        )
        # load and update image paths
        coco_json_path = os.path.join(tmp_dir, "result.json")
        coco_annotations = load_json(coco_json_path)
        # images_metadata = coco_annotations['images']
        # images_metadata_updated = list()
        # for metadata in tqdm(images_metadata,desc=f'json-to-coco for {input_file}'):
        #     if metadata['width'] is not None:
        #         # print(metadata['file_name'])
        #         # file_path = get_local_path(metadata['file_name'])
        #         file_path = metadata['file_name']
        #         metadata_copy = copy(metadata)
        #         metadata_copy.update({'file_name':file_path})
        #         images_metadata_updated.append(metadata_copy)
        # coco_annotations.update({'images':images_metadata_updated})
        # save if requested
        if out_file_name is not None:
            with open(out_file_name, "w") as file:
                json.dump(coco_annotations, file, indent=2)

        return coco_annotations


def get_upload_img_dir(coco_annotation: dict):
    directory = list(
        set(
            [
                os.path.dirname(metadata["file_name"])
                for metadata in coco_annotation["images"]
            ]
        )
    )

    if len(directory) > 1:
        print(
            f"There should be one upload directory per annotation project. There are {len(directory)}={directory}. Attempting to fix it through os.path.commonprefix. Not guaranteed to work."
        )
        return os.path.commonprefix(directory)

    if len(directory) == 0:
        raise NotImplementedError("There are no labels")

    return directory.pop()


def convert_json_annotations_to_coco(
    input_dir: str,
    dest_dir_coco: str,
    parse_ls_config: bool = False,
    dotenv_path: str = None,
    ls_client: Client = None,
    ls_xml_config: str = None,
) -> dict:
    """Converts directory with LS json files to coco format.

    Args:
        input_dir (str, optional): directory with LS json annotation files. Defaults to JSON_DIR_PATH.
        dest_dir_coco (str, optional): destination directory. It should be different from input_dir. Defaults to COCO_DIR_PATH.

    Returns:
        dict: the schema is {uploaded_image_dir:coco_annotation_path}
    """

    def get_ls_parsed_config(ls_json_path: str):
        if dotenv_path is not None:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path)

        labelstudio_client = ls_client
        if ls_client is None:
            # Connect to the Label Studio API and check the connection
            LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
            API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
            labelstudio_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

        with open(ls_json_path, "r") as f:
            ls_annotation = json.load(fp=f)
        ids = set([annot["project"] for annot in ls_annotation])
        assert len(ids) == 1, "annotations come from different project. Not allowed!"
        project_id = ids.pop()
        project = labelstudio_client.get_project(id=project_id)

        return project.parsed_label_config

    upload_img_dirs, coco_paths = list(), list()

    for path in Path(input_dir).glob("*.json"):
        coco_path = os.path.join(dest_dir_coco, path.name)
        parsed_config = None
        if parse_ls_config:
            parsed_config = get_ls_parsed_config(path)
        annot = convert_json_to_coco(
            path,
            out_file_name=coco_path,
            parsed_config=parsed_config,
            ls_xml_config=None,
        )
        upload_img_dirs.append(get_upload_img_dir(coco_annotation=annot))
        coco_paths.append(coco_path)

    return dict(zip(upload_img_dirs, coco_paths))


def load_coco_annotations(dest_dir_coco: str) -> dict:
    """Loads existing coco annotations

    Args:
        dest_dir_coco (str, optional): directory with annotations in coco format.

    Returns:
        dict: the schema is {uploaded_image_dir:coco_annotation_path}
    """

    coco_paths = list(Path(dest_dir_coco).glob("*.json"))
    upload_img_dirs = [
        get_upload_img_dir(coco_annotation=load_json(coco_path))
        for coco_path in coco_paths
    ]

    return dict(zip(upload_img_dirs, coco_paths))


def get_slices(
    coco_annotation_file_path: str,
    img_dir: str,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    slice_height: int = 640,
    slice_width: int = 640,
    min_area_ratio: float = 0.1,
    ignore_negative_samples: bool = False,
    verbose: bool = False,
) -> dict:
    """Slices annotations. This function call slice_coco from ``sahi``. See https://github.com/obss/sahi/blob/main/sahi/slicing.py#L413

    Args:
        coco_annotation_file_path (str): path to annotation
        img_dir (str): directory of images
        overlap_height_ratio (float, optional): Defaults to 0.2.
        overlap_width_ratio (float, optional): Defaults to 0.2.
        slice_height (int, optional): Defaults to 640.
        slice_width (int, optional): Defaults to 640.
        min_area_ratio (float, optional):If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Defaults to 0.1.
        ignore_negative_samples (bool, optional): states if empty images should be ignored. Defaults to False.
        verbose (bool, optional): Defaults to False.

    Returns:
        dict: sliced annotations
    """

    # print(coco_annotation_file_path)
    sliced_coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=img_dir,
        output_coco_annotation_file_name=None,  # os.path.join(TEMP, "sliced_coco.json"),
        ignore_negative_samples=ignore_negative_samples,
        # output_dir="",
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        min_area_ratio=min_area_ratio,
        verbose=verbose,
        out_ext=".jpg",
    )

    return sliced_coco_dict


def sample_data(
    coco_dict_slices: dict,
    img_dir: str,
    empty_ratio: float = 3.0,
    save_all: bool = False,
    out_csv_path: str = None,
    labels_to_discard: list = None,
    labels_to_keep: list = None,
    sample_only_empty: bool = False,
) -> pd.DataFrame:
    """Sample annotations from sliced coco annotations

    Args:
        coco_dict_slices (dict): sliced coco annotation from .utils.get_slices
        img_dir (str): image directory
        empty_ratio (float, optional): ratio negative samples (i.e. empty images) to positive samples. Defaults to 3. It loads 3 times more empty tiles than non-empty.
        out_csv_path (str, optional): if given, it saves the sampled annotations to the path. Defaults to None.
        labels_to_discard (list, optional): labels to discard. Defaults to None.
        labels_to_keep (list, optional): labels to keep. Defaults.
        sample_only_empty (bool, optional): states if only negative samples should be saved. It is useful for hard negative sample mining. Defaults to False.

    Raises:
        FileNotFoundError: raised when a parent image can't be found for a sliced_image (i.e. tile)

    Returns:
        pd.DataFrame: sampled annotations. Columns are 'x' (center), 'y' (center), 'w', 'h', 'x_min', 'y_min', 'x_max', 'y_max' etc.
    """

    assert empty_ratio >= 0.0, "Provide appropriate value"
    assert (save_all + sample_only_empty) < 2, "Both cannot be true."
    assert (labels_to_discard is not None) + (labels_to_keep is not None) <= 1, (
        "At most one should be given!"
    )

    def get_parent_image(file_name: str):
        ext = ".jpg"
        file_name = Path(file_name).stem
        # print(file_name)
        parent_file = "_".join(file_name.split("_")[:-5])
        p = os.path.join(img_dir, parent_file + ext)
        if os.path.exists(p):
            return p
        raise FileNotFoundError(
            f"Parent file note found for {file_name} in {img_dir} >> {parent_file}"
        )

    # build mapping for labels
    label_ids = [cat["id"] for cat in coco_dict_slices["categories"]]
    label_name = [cat["name"] for cat in coco_dict_slices["categories"]]
    label_map = dict(zip(label_ids, label_name))

    # build dataFrame of image slices
    ids = list()
    x0s, x1s = list(), list()
    y0s, y1s = list(), list()
    file_paths = list()
    parent_file_paths = list()
    for metadata in coco_dict_slices["images"]:
        # img_path = os.path.join(img_dir,metadata['file_name'])
        file_paths.append(metadata["file_name"])
        file_name = os.path.basename(metadata["file_name"])
        x_0, y_0, x_1, y_1 = file_name.split(".")[0].split("_")[-4:]
        parent_image = get_parent_image(file_name)
        parent_file_paths.append(parent_image)
        x0s.append(int(x_0))
        x1s.append(int(x_1))
        y0s.append(int(y_0))
        y1s.append(int(y_1))
        ids.append(metadata["id"])
    df_limits = {
        "x0": x0s,
        "x1": x1s,
        "y0": y0s,
        "y1": y1s,
        "id": ids,
        "images": file_paths,
        "parent_images": parent_file_paths,
    }
    df_limits = pd.DataFrame.from_dict(df_limits, orient="columns")
    df_limits.set_index("id", inplace=True)

    # build dataframe of annotations
    x_mins, y_mins = list(), list()
    widths, heights = list(), list()
    ids_annot = list()
    label_ids = list()
    for annot in coco_dict_slices["annotations"]:
        ids_annot.append(annot["image_id"])
        x, y, w, h = annot["bbox"]
        label_ids.append(annot["category_id"])
        x_mins.append(x)
        y_mins.append(y)
        widths.append(w)
        heights.append(h)
    df_annot = {
        "x_min": x_mins,
        "y_min": y_mins,
        "width": widths,
        "height": heights,
        "id": ids_annot,
        "label_id": label_ids,
    }
    df_annot = pd.DataFrame.from_dict(df_annot, orient="columns")
    df_annot.set_index("id", inplace=True)
    df_annot["labels"] = df_annot["label_id"].map(label_map)
    for col in ["x_min", "y_min", "width", "height"]:
        df_annot.loc[:, col] = df_annot[col].apply(math.floor)

    # join dataframes
    df = df_limits.join(df_annot, how="outer")

    # get non-empty
    df_empty = df[df["x_min"].isna()].copy()
    df_empty.drop_duplicates(subset="images", inplace=True)

    # discard non-animal labels
    if labels_to_discard is not None:
        df = df[~df.labels.isin(labels_to_discard)].copy()
        df_non_empty = df[~df["x_min"].isna()].copy()
    elif labels_to_keep is not None:
        df_non_empty = df[df.labels.isin(labels_to_keep)].copy()
    else:
        df = df[~df.labels.isin(labels_to_discard)].copy()
        print(
            "sample_data function: No label is discarded, they are all kept", end="\n"
        )

    # get number of images to sample
    non_empty_num = df_non_empty["images"].unique().shape[0]
    empty_num = math.floor(non_empty_num * empty_ratio)
    empty_num = min(empty_num, len(df_empty))
    frac = 1.0 if save_all else empty_num / len(df_empty)

    # get empty df and tiles
    if sample_only_empty:
        df = df_empty.sample(frac=empty_num / len(df_empty))
        df.reset_index(inplace=True)
        # create x_center and y_center
        df["x"] = np.nan
        df["y"] = np.nan
        df["width"] = np.nan
        df["height"] = np.nan
    else:
        df_empty = df_empty.sample(frac=frac, random_state=41, replace=False)
        print(
            f"Sampling {len(df_empty)} empty images, and {non_empty_num} non-empty images."
        )

        # concat dfs
        df = pd.concat([df_empty, df_non_empty], axis=0)
        df.reset_index(inplace=True)

        # create x_center and y_center
        df["x"] = df["x_min"] + df["width"] * 0.5
        df["y"] = df["y_min"] + df["height"] * 0.5

    # save df
    if out_csv_path is not None:
        df.to_csv(out_csv_path, sep=",", index=False)

    return df


def save_tiles(
    df_tiles: pd.DataFrame, out_img_dir: str, clear_out_img_dir: bool = False
) -> None:
    """Saves tiles (or slices of images) as .jpg files

    Args:
        df_tiles (pd.DataFrame): provides tiles boundaries. Computed from .utils.sample_data
        out_img_dir (str): output directory to save tiles
        clear_out_img_dir (bool, optional): states if output directory should be emptied. Defaults to False.
    """

    # clear out_img_dir
    if clear_out_img_dir:
        print("Deleting images in ", out_img_dir)
        shutil.rmtree(out_img_dir)
        Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    # selecting non-duplicated
    df_tiles = df_tiles[~df_tiles.duplicated(["x0", "x1", "y0", "y1", "images"])].copy()

    for idx in tqdm(df_tiles.index, desc=f"Saving tiles to {out_img_dir}"):
        x0 = int(df_tiles.at[idx, "x0"])
        x1 = int(df_tiles.at[idx, "x1"])
        y0 = int(df_tiles.at[idx, "y0"])
        y1 = int(df_tiles.at[idx, "y1"])
        img_path = df_tiles.at[idx, "parent_images"]
        tile_name = df_tiles.at[idx, "images"]
        save_path = str(Path(os.path.join(out_img_dir, tile_name)).with_suffix(".jpg"))
        img = imread(img_path)
        tile = img[y0:y1, x0:x1, :]
        imsave(fname=save_path, arr=tile, check_contrast=False)


def load_label_map(
    path: str, label_to_discard: list = None, labels_to_keep: list = None
) -> dict:
    """Loads label map. The map should be a json mapping class index to class name.

    Args:
        path (str): path to json file
        label_to_discard (list): labels to discard. If none, proide an empty list
        label_to_keep (list): should be given when label_to_discard is not

    Returns:
        dict: label map {index:name}
    """

    assert (labels_to_keep is not None) + (label_to_discard is not None) == 1, (
        "Exactly one should be None."
    )

    # load label mapping
    with open(path, "r") as file:
        label_map = json.load(file)
    if label_to_discard is not None:
        names = [p["name"] for p in label_map if p["name"] not in label_to_discard]
    else:
        names = [p["name"] for p in label_map if p["name"] in labels_to_keep]
    label_map = dict(zip(range(len(names)), names))
    return label_map


def update_yolo_data_cfg(data_config_yaml, label_map: dict):
    """Updates yolo data config yaml file "names" and "nc" fields.

    Args:
        args (Dataprepconfigs): configs
    """

    # assert args.label_map is not None, 'Provide path to label mapping.'

    # load yaml
    with open(data_config_yaml, "r") as file:
        yolo_config = yaml.load(file, Loader=yaml.FullLoader)
    # load label mapping
    # updaate yaml and save
    yolo_config.update({"names": label_map, "nc": len(label_map)})
    with open(data_config_yaml, "w") as file:
        yaml.dump(yolo_config, file, default_flow_style=False, sort_keys=False)


def save_df_as_yolo(
    df_annotation: pd.DataFrame,
    dest_path_labels: str,
    slice_width: int,
    slice_height: int,
):
    """Saves annotations as yolo dataset.

    Args:
        df_annotation (pd.DataFrame): annotations.
            Should have the columns ['label_id','x','y','width','height'] with no NaN values.
            The bounbing box cordinates are expected to be in sliced imaged coordinate system.
        dest_path_labels (str): destination
        slice_width (int): image slice width to be used for normalization
        slice_height (int): image slice height to be used for normalization
    """

    cols = ["label_id", "x", "y", "width", "height"]
    for col in cols:
        assert df_annotation[col].isna().sum() < 1, "there are NaN values. Check out."

    # change type
    for col in cols[1:]:
        df_annotation.loc[:, col] = df_annotation[col].apply(float)
    df_annotation.loc[:, "label_id"] = df_annotation["label_id"].apply(int)
    df_annotation = df_annotation.astype({"label_id": "int32"})

    # normalize values
    df_annotation.loc[:, "x"] = df_annotation["x"].apply(lambda x: x / slice_width)
    df_annotation.loc[:, "y"] = df_annotation["y"].apply(lambda y: y / slice_height)
    df_annotation.loc[:, "width"] = df_annotation["width"].apply(
        lambda x: x / slice_width
    )
    df_annotation.loc[:, "height"] = df_annotation["height"].apply(
        lambda y: y / slice_height
    )

    # check value range
    assert df_annotation[cols[1:]].all().max() <= 1.0, "max value <= 1"
    assert df_annotation[cols[1:]].all().min() >= 0.0, "min value >=0"

    for image_name, df in tqdm(
        df_annotation.groupby("images"), desc="Saving yolo labels"
    ):
        txt_file = image_name.split(".")[0] + ".txt"
        df[cols].to_csv(
            os.path.join(dest_path_labels, txt_file), sep=" ", index=False, header=False
        )


def build_yolo_dataset(args: Dataprepconfigs):
    """Builds a yolo dataset

    Args:
        args (Dataprepconfigs): arguments defining desired behavior. See tutorials for a better explanation.

    """

    # Checking inconsistency in arguments
    if (args.clear_yolo_dir + args.load_coco_annotations) == 2:
        raise ValueError(
            "Warning : both clear_yolo_dir and load_coco_annotations are enabled! it is likely to not work as expected."
        )

    # clear directories
    if args.clear_yolo_dir:
        for p in [args.dest_path_images, args.dest_path_labels, args.coco_json_dir]:
            shutil.rmtree(p)
            Path(p).mkdir(parents=True, exist_ok=True)

    # convert ls json to coco
    if args.load_coco_annotations:
        map_imgdir_cocopath = load_coco_annotations(dest_dir_coco=args.coco_json_dir)
    else:
        map_imgdir_cocopath = convert_json_annotations_to_coco(
            input_dir=args.ls_json_dir,
            dest_dir_coco=args.coco_json_dir,
            parse_ls_config=args.parse_ls_config,
        )

    # load label map and update yolo data_cfg_yaml file
    if not args.is_detector:
        label_map = load_label_map(
            path=args.label_map,
            label_to_discard=args.discard_labels,
            labels_to_keep=args.keep_labels,
        )
        update_yolo_data_cfg(args.data_config_yaml, label_map=label_map)
        name_id_map = {val: key for key, val in label_map.items()}

    # slice coco annotations and save tiles
    for img_dir, cocopath in tqdm(
        map_imgdir_cocopath.items(), desc="Building yolo dataset"
    ):
        try:
            # slice annotations
            coco_dict_slices = get_slices(
                coco_annotation_file_path=cocopath,
                img_dir=img_dir,
                slice_height=args.height,
                slice_width=args.width,
                overlap_height_ratio=args.overlap_ratio,
                overlap_width_ratio=args.overlap_ratio,
                min_area_ratio=args.min_visibility,
                ignore_negative_samples=(
                    args.empty_ratio < 1e-8 and not args.save_all
                ),  # equivalent to args.empty_ratio == 0.0
            )
            # sample tiles
            df_tiles = sample_data(
                coco_dict_slices=coco_dict_slices,
                empty_ratio=args.empty_ratio,
                out_csv_path=None,  # Path(args.dest_path_images).with_name("gt.csv"),
                img_dir=img_dir,
                save_all=args.save_all,
                labels_to_discard=args.discard_labels,
                labels_to_keep=args.keep_labels,
                sample_only_empty=args.save_only_empty,
            )

            # detector_training mode
            if args.is_detector:
                df_tiles["label_id"] = 0
            else:
                df_tiles["label_id"] = df_tiles["labels"].map(name_id_map)
                mask = ~df_tiles["label_id"].isna()
                df_tiles.loc[mask, "label_id"] = df_tiles.loc[mask, "label_id"].apply(
                    int
                )
                # raise NotImplementedError('Pipeline not designed to handle multiple classes.')

            # save labels in yolo format
            save_df_as_yolo(
                df_annotation=df_tiles.dropna(axis=0, how="any"),
                slice_height=args.height,
                slice_width=args.width,
                dest_path_labels=args.dest_path_labels,
            )
            # save tiles
            save_tiles(
                df_tiles=df_tiles,
                out_img_dir=args.dest_path_images,
                clear_out_img_dir=False,
            )
        except Exception:
            print("--" * 25, end="\n")
            traceback.print_exc()
            print("--" * 25)
            print(f"Failed to build yolo dataset for for {img_dir} -- {cocopath}\n\n")
