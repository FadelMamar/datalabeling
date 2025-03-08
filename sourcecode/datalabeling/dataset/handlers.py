from animaloc.datasets import CSVDataset



class HerdnetPredictDataset(CSVDataset):

    def __init__(self, images_path:list[str], albu_transforms = None, end_transforms = None):

        assert isinstance(images_path,list)

        images_path = list(map(str,images_path))
        # create dummy df_labels
        num_images = len(images_path)
        df_labels = {'x':[0.]*num_images,
                     'y':[0.]*num_images,
                     'labels':[0]*num_images,
                     'images':images_path
                     }
        df_labels = pd.DataFrame.from_dict(df_labels)
        super().__init__(csv_file=df_labels, 
                         root_dir="", 
                         albu_transforms=albu_transforms,
                         end_transforms=end_transforms)
    
    def _load_image(self, index: int) -> Image.Image:
        img_name = self._img_names[index]
        return Image.open(img_name).convert('RGB')
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:        
        img = self._load_image(index)
        target = self._load_target(index)
        tr_img, tr_target = self._transforms(img, target)

        return tr_img