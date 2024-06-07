# imports
from pathlib import Path
import json
from typing import List,Dict
from label_studio_ml.utils import get_env
import math
import pandas as pd
from arguments import Arguments
import os
import PIL
import torchvision
import numpy
import cv2
import math
from tqdm import tqdm
import urllib
from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images
from albumentations import PadIfNeeded


# paths
JSON_DIR_PATH = "../exported_annotations/json"
CSV_DIR_PATH = "../exported_annotations/csv"
ALL_CSV = "../exported_annotations/all_csv.csv"

def get_local_path(url:str):
    filename, dir_path = url.split('/data/', 1)[-1].split('?d=')
    dir_path = str(urllib.parse.unquote(dir_path))
    LOCAL_FILES_DOCUMENT_ROOT = get_env(
        'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT', default=os.path.abspath(os.sep)
    )
    filepath = os.path.join(LOCAL_FILES_DOCUMENT_ROOT,
                            dir_path) #.replace('C:','D:')
    
    return filepath

def load_ls_annotations():

    ls_annotations = []
    paths = list(Path(JSON_DIR_PATH).glob('*.json'))

    for json_file in paths:
        with open(json_file,'r') as f:
            annotation = json.load(fp=f)
            ls_annotations.append(annotation)

    return ls_annotations,paths

def convert_json_to_df(json_data:List[Dict]):

    # Reference for label studio output schema  https://labelstud.io/tags/rectanglelabels

    # placeholders
    image_paths = list()
    xs = list()
    ys = list()
    widths = list()
    heights = list()
    labels = list()
    
    for data in json_data:
        # load data
        if 'label' in data.keys():
            image_path = get_local_path(url=data['image'])
            for rectangle in data['label']:
                x = math.ceil(rectangle['x']*rectangle['original_width']/100)
                y = math.ceil(rectangle['y']*rectangle['original_height']/100)
                w = math.ceil(rectangle['width']*rectangle['original_width']/100)
                h = math.ceil(rectangle['height']*rectangle['original_height']/100)
                label = rectangle['rectanglelabels'][0]
                # update placeholders
                image_paths.append(image_path)
                xs.append(x)
                ys.append(y)
                widths.append(w)
                heights.append(h)
                labels.append(label)
    
    # csv data
    csv_data = {'images':image_paths,
                'x':xs,
                'y':ys,
                'width':widths,
                'height':heights,
                'labels':labels
                }
    df = pd.DataFrame.from_dict(data=csv_data,orient='columns')
    df['x_min'] = df['x'].copy()
    df['x_max'] = df['x'] + df['width'].apply(int)
    df['y_min'] = df['y'].copy()
    df['y_max'] = df['y'] + df['height'].apply(int)
    df['x'] = df['x_min'] + (df['width']*0.5).apply(int)
    df['y'] = df['y_min'] + (df['height']*0.5).apply(int)

    return df

def convert_json_annotations_to_csv(rewrite_existing=True):

    # load all available annotations
    annotations, paths = load_ls_annotations()

    # parse and save annotations
    for annotation, path in tqdm(zip(annotations,paths),desc='Converting jsons to csv'):
        save_path = f"../exported_annotations/csv/{path.name.removesuffix('.json')}.csv"
        if Path(save_path).exists():
            if rewrite_existing:
                df = convert_json_to_df(json_data=annotation)
                df.to_csv(save_path,index=False)
        else:
            df = convert_json_to_df(json_data=annotation)
            df.to_csv(save_path,index=False)

def save_df_as_yolo(df_annotation:pd.DataFrame,args:Arguments):
    
    cols = ['labels','x','y','width','height']
    df_annotation['x'] = df_annotation['x']/args.width
    df_annotation['y'] = df_annotation['y']/args.height
    df_annotation['width'] = df_annotation['width']/args.width
    df_annotation['height'] = df_annotation['height']/args.height
    
    for image_name,df in df_annotation.groupby('images'):
        txt_file = image_name.split('.')[0] + '.txt'
        df[cols].to_csv(os.path.join(args.dest_path_labels,txt_file),sep=' ',index=False,header=False)

def patcher(args:Arguments):
    
    # creating destination directory for tiles
    Path(args.dest_path_images).mkdir(parents=True,exist_ok=True)
    Path(args.dest_path_labels).mkdir(parents=True,exist_ok=True)
    
    # get images paths 
    images_paths = set()
    dfs = list()
    for csv_path in Path(CSV_DIR_PATH).glob('*.csv'):
        df = pd.read_csv(csv_path,sep=',')
        images_paths = images_paths.union(set(df['images'].to_list()))
        dfs.append(df)

    # merge all csv and save
    dfs_concat = pd.concat(dfs,axis=0)

    # discard non-animals labels
    dfs_concat = dfs_concat[~dfs_concat.labels.isin(args.discard_labels)]

    # encode to numerical values
    if args.is_detector:
        dfs_concat['labels'] = 0
    else:
        raise NotImplementedError
    dfs_concat.to_csv(ALL_CSV,index=False)

    # tile and save
    if ALL_CSV is not None:
        patches_buffer = PatchesBuffer(ALL_CSV, args.root_path,
                                       (args.height, args.width),
                                       overlap=args.overlap,
                                       min_visibility=args.min_visibility).buffer
        patches_buffer['images'] = patches_buffer['images'].apply(os.path.basename)
        patches_buffer.drop(columns='limits').to_csv(os.path.join(args.dest_path_labels, 'gt.csv'),
                                                     index=False)
        patches_buffer['base_images'] = patches_buffer['base_images'].apply(os.path.basename)
        
        # save labels in yolo format
        save_df_as_yolo(df_annotation=patches_buffer, args=args)
    
    for img_path in tqdm(images_paths, desc='Exporting patches'):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)
        
        if ALL_CSV is not None:
            # save all patches
            if args.save_all:
                patches = ImageToPatches(img_tensor, (args.height, args.width),
                                         overlap=args.overlap).make_patches()
                save_batch_images(patches, img_name, args.dest_path_images)
            # or only annotated ones
            else:
                padder = PadIfNeeded(
                    args.height, args.width,
                    position = PadIfNeeded.PositionType.TOP_LEFT,
                    border_mode = cv2.BORDER_CONSTANT,
                    value= 0
                    )
                img_ptch_df = patches_buffer[patches_buffer['base_images']==img_name]
                for row in img_ptch_df[['images','limits']].to_numpy().tolist():
                    ptch_name, limits = row[0], row[1]
                    cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                    padded_img = PIL.Image.fromarray(padder(image = cropped_img)['image'])
                    padded_img.save(os.path.join(args.dest_path_images, ptch_name))
        else:
            patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
            save_batch_images(patches, img_name, args.dest_path_images)



# Demo
if __name__ == '__main__':

    from arguments import Arguments
    convert_json_annotations_to_csv()
    args = Arguments()
    patcher(args=args)

    pass