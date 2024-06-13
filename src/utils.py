# imports
from pathlib import Path
from copy import copy
import io
import json
from typing import List,Dict
from label_studio_ml.utils import get_env ,get_local_path
from sahi.utils.file import load_json
from sahi.slicing import slice_coco
from skimage.io import imread,imsave
import shutil
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
from label_studio_converter import Converter

# paths
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR_PATH = os.path.join(CUR_DIR,"../exported_annotations/json")
JSONMIN_DIR_PATH = os.path.join(CUR_DIR,"../exported_annotations/json-min")
CSV_DIR_PATH = os.path.join(CUR_DIR,"../exported_annotations/csv")
COCO_DIR_PATH = os.path.join(CUR_DIR,"../exported_annotations/coco-format")
ALL_CSV = os.path.join(CUR_DIR,"../exported_annotations/all_csv.csv")
LABELSTUDIOCONFIG = os.path.join(CUR_DIR,"../exported_annotations/label_studio_config.xml")
TEMP = os.path.join(CUR_DIR,"../.tmp")


# def get_local_path(url:str):
#     filename, dir_path = url.split('/data/', 1)[-1].split('?d=')
#     dir_path = str(urllib.parse.unquote(dir_path))
#     LOCAL_FILES_DOCUMENT_ROOT = get_env(
#         'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT', default=os.path.abspath(os.sep)
#     )
#     filepath = os.path.join(LOCAL_FILES_DOCUMENT_ROOT,
#                             dir_path) #.replace('C:','D:')
    
#     return filepath

def load_ls_annotations(input_dir:str=JSONMIN_DIR_PATH):

    ls_annotations = []
    paths = list(Path(input_dir).glob('*.json'))

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
    image_widths = list()
    image_heights = list()
    
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
                image_heights.append(data['label'][0]['original_height'])
                image_widths.append(data['label'][0]['original_width'])
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
                'labels':labels,
                'image_height':image_heights,
                'image_width':image_widths
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
    annotations, paths = load_ls_annotations(input_dir=JSONMIN_DIR_PATH)

    # parse and save annotations
    for annotation, path in tqdm(zip(annotations,paths),desc='Converting jsons to csv'):
        save_path = os.path.join(CSV_DIR_PATH,path.name.replace('.json','.csv'))
        if Path(save_path).exists():
            if rewrite_existing:
                df = convert_json_to_df(json_data=annotation)
                df.to_csv(save_path,index=False)
        else:
            df = convert_json_to_df(json_data=annotation)
            df.to_csv(save_path,index=False)

def convert_json_to_coco(input_file:str,out_file_name:str=None):
    
    # load converter
    with io.open(LABELSTUDIOCONFIG) as f:
        config_str = f.read()
    handler = Converter(config=config_str,
                        project_dir=None,
                        download_resources=False)
    handler.convert_to_coco(input_data=input_file,
                            output_dir=TEMP,
                            output_image_dir=os.path.join(TEMP,'images'),
                            is_dir=False)
    # load and update image paths
    coco_json_path = os.path.join(TEMP,'result.json')
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
        with open(out_file_name,'w') as file:
            json.dump(coco_annotations,file,indent=2)  

    return coco_annotations

def convert_json_annotations_to_coco(input_dir:str=JSON_DIR_PATH)->dict:

    def get_upload_img_dir(coco_annotation:dict):
        directory = set([os.path.dirname(metadata['file_name']) for metadata in coco_annotation['images']])
        assert len(directory)==1,'There should be one upload directory per annotation project'
        return directory.pop() #list(directory)[0]

    upload_img_dirs,coco_paths = list(),list()
    for path in Path(input_dir).glob('*.json'):
        coco_path = os.path.join(COCO_DIR_PATH,path.name)
        annot = convert_json_to_coco(path,out_file_name=coco_path)
        upload_img_dirs.append(get_upload_img_dir(coco_annotation=annot))
        coco_paths.append(coco_path)

    return dict(zip(upload_img_dirs,coco_paths))
    
def get_slices(coco_annotation_file_path:str,img_dir:str,
               overlap_height_ratio:float=0.2,overlap_width_ratio:float=0.2,
               slice_height:int=640,slice_width:int=640,
               min_area_ratio:float=0.1,
               ignore_negative_samples:bool=False,
               verbose:bool=False)->dict:
    # print(coco_annotation_file_path)
    sliced_coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=img_dir,
    output_coco_annotation_file_name=os.path.join(TEMP,"sliced_coco.json"),
    ignore_negative_samples=ignore_negative_samples,
    # output_dir="",
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    min_area_ratio=min_area_ratio,
    verbose=verbose,
    out_ext='.jpg',
    )

    return sliced_coco_dict

def sample_data(coco_dict_slices:dict,
                img_dir:str,
                empty_ratio:int=3,
                out_csv_path:str=None,
                labels_to_discard:list=None)->pd.DataFrame:
    
    assert (empty_ratio >= 0) and isinstance(empty_ratio,int),'Provide appropriate value'

    def get_parent_image(file_name:str,lookup_extensions:list[str] = ['.jpg','.png','.tif','.JPG','.PNG','.TIF']):
        ext = '.jpg' #Path(file_name).suffix
        file_name = Path(file_name).stem
        # print(file_name)
        parent_file = '_'.join(file_name.split('_')[:-5])
        # for ext in lookup_extensions:
        p = os.path.join(img_dir,parent_file+ext)
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f'Parent file note found for {file_name} in {img_dir} >> {parent_file}')
    
    
    # build mapping for labels
    label_ids = [cat['id'] for cat in coco_dict_slices['categories']]
    label_name = [cat['name'] for cat in coco_dict_slices['categories']]
    label_map = dict(zip(label_ids,label_name))

    # build dataFrame of image slices 
    ids = list()
    x0s, x1s = list(), list()
    y0s, y1s = list(), list()
    file_paths = list()
    parent_file_paths = list()
    for metadata in coco_dict_slices['images']:
        # img_path = os.path.join(img_dir,metadata['file_name'])
        file_paths.append(metadata['file_name'])
        file_name = os.path.basename(metadata['file_name'])
        x_0,y_0,x_1,y_1 = file_name.split('.')[0].split('_')[-4:]
        parent_image = get_parent_image(file_name)
        parent_file_paths.append(parent_image)
        x0s.append(x_0)
        x1s.append(x_1)
        y0s.append(y_0)
        y1s.append(y_1)
        ids.append(metadata['id'])
    df_limits = {'x0':x0s,
                 'x1':x1s,
                 'y0':y0s,
                 'y1':y1s,
                 'id':ids,
                 'images':file_paths,
                 'parent_images':parent_file_paths
                 }
    df_limits = pd.DataFrame.from_dict(df_limits,orient='columns')
    df_limits.set_index('id',inplace=True)

    # build dataframe of annotations
    x_mins, y_mins = list(), list()
    widths, heights = list(), list()
    ids_annot = list()
    label_ids = list()
    for annot in coco_dict_slices['annotations']:
        ids_annot.append(annot['image_id'])
        x,y,w,h = annot['bbox']
        label_ids.append(annot['category_id'])
        x_mins.append(x)
        y_mins.append(y)
        widths.append(w)
        heights.append(h)
    df_annot = {'x_min':x_mins,
                'y_min':y_mins,
                'width':widths,
                'height':heights,
                'id':ids_annot,
                'label_id':label_ids}
    df_annot = pd.DataFrame.from_dict(df_annot,orient='columns') 
    df_annot.set_index('id',inplace=True)
    df_annot['labels'] = df_annot['label_id'].map(label_map)
    for col in ['x_min','y_min','width','height']:
        df_annot[col] = df_annot[col].apply(math.floor)

    # join dataframes
    df = df_limits.join(df_annot,how='outer') 

    # get empty df and tiles
    df_empty = df[df['x_min'].isna()]
    df_non_empty = df[~df['x_min'].isna()]
    empty_num =  int(len(df_non_empty)*empty_ratio)
    df_empty = df_empty.sample(n=empty_num,random_state=41,replace=False)

    # concat dfs
    df = pd.concat([df_empty,df[~df['x_min'].isna()]],axis=0)
    df.reset_index(inplace=True)

    # discard non-animal labels
    if labels_to_discard is not None:
        df = df[~df.labels.isin(labels_to_discard)]


    # create x_center and y_center
    df['x'] = df['x_min'] + df['width']*0.5
    df['y'] = df['y_min'] + df['height']*0.5

    # save df
    if out_csv_path is not None:
        df.to_csv(out_csv_path,sep=',',index=False)

    return df

def save_tiles(df_tiles:pd.DataFrame,out_img_dir:str,clear_out_img_dir:bool=False)->None:

    # clear out_img_dir
    if clear_out_img_dir:
        print('Deleting images in ',out_img_dir)
        shutil.rmtree(out_img_dir)
        Path(out_img_dir).mkdir(parents=True,exist_ok=True) 

    for idx in tqdm(df_tiles.index,desc=f'Saving tiles to {out_img_dir}'):
        x0 = int(df_tiles.at[idx,'x0'])
        x1 = int(df_tiles.at[idx,'x1'])
        y0 = int(df_tiles.at[idx,'y0'])
        y1 = int(df_tiles.at[idx,'y1'])
        img_path = df_tiles.at[idx,'parent_images']
        tile_name = df_tiles.at[idx,'images']
        img = imread(img_path)
        tile = img[y0:y1,x0:x1,:]
        imsave(fname=os.path.join(out_img_dir,tile_name),arr=tile)

def save_df_as_yolo(df_annotation:pd.DataFrame,dest_path_labels:str,slice_width:int,slice_height:int):
    
    cols = ['labels','x','y','width','height']
    for col in cols:
        assert df_annotation[col].isna().sum()<1,'there are NaN values. Check out.'

    df_annotation['x'] = df_annotation['x']/slice_width
    df_annotation['y'] = df_annotation['y']/slice_height
    df_annotation['width'] = df_annotation['width']/slice_width
    df_annotation['height'] = df_annotation['height']/slice_height
    
    for image_name,df in df_annotation.groupby('images'):
        txt_file = image_name.split('.')[0] + '.txt'
        df[cols].to_csv(os.path.join(dest_path_labels,txt_file),sep=' ',index=False,header=False)

def build_yolo_dataset(args:Arguments,ls_json_dir:str=JSON_DIR_PATH,clear_out_dir:bool=False):

    #clear directories
    if clear_out_dir:
        for p in [args.dest_path_images,args.dest_path_labels,COCO_DIR_PATH]:
            shutil.rmtree(p)
            Path(p).mkdir(parents=True,exist_ok=True)

    # convert ls json to coco
    map_imgdir_cocopath = convert_json_annotations_to_coco(input_dir=ls_json_dir)

    # slice coco annotations and save tiles
    for img_dir,cocopath in map_imgdir_cocopath.items():
        # slice annotations
        coco_dict_slices = get_slices(coco_annotation_file_path=cocopath,
                            img_dir=img_dir,
                            slice_height=args.height,
                            slice_width=args.width,
                            overlap_height_ratio=args.overlap_ratio,
                            overlap_width_ratio=args.overlap_ratio,
                            min_area_ratio=args.min_visibility
                            )
        # sample tiles
        df_tiles = sample_data(coco_dict_slices=coco_dict_slices,
                                empty_ratio=args.empty_ratio,
                                out_csv_path=ALL_CSV,
                                img_dir=img_dir,
                                labels_to_discard=args.discard_labels
                                )
        # save tiles
        save_tiles(df_tiles=df_tiles,
                   out_img_dir=args.dest_path_images,
                   clear_out_img_dir=False)
        # save labels in yolo format
        save_df_as_yolo(df_annotation=df_tiles[~df_tiles['x'].isna()].copy(),
                        slice_height=args.height,
                        slice_width=args.width,
                        dest_path_labels=args.dest_path_labels)

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
# if __name__ == '__main__':
#     args = Arguments()
# #     build_yolo_dataset(args=args,clear_out_dir=True)
#     # patcher(args=args)
#     # convert_json_to_coco(input_file=r"..\exported_annotations\json\project-1-at-2024-06-09-00-41-b6d95d93.json")
#     pass