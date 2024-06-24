from pathlib import Path

def main(dataset_dir:Path):

    save_directory = dataset_dir/'preprocessed'
    save_directory.mkdir(parents=False,exist_ok=True)

    for img_path in Path(dataset_dir).iterdir():
        try:
            data = imread(str(img_path))
            tilesize_h = 1000
            tilesize_w = 1000
            height, width, channels = data.shape 
            for i,j in tqdm(product(list(range(0,height,tilesize_h)),list(range(0,width,tilesize_w)))):
                tile = data[i:min(i+tilesize_h,height),j:min(j+tilesize_w,width),:]
                filename = img_path.name.split('.')[0] + f'#{i}#{j}' + img_path.suffix
                savepath = save_directory/filename
                imsave(savepath,tile)
        except Exception as e:
            print(e)
            print(f"Failed for: {img_path}\n")


if __name__ == '__main__':
    #TODO: change path to directory
    dataset_dir = Path(r"C:\Users\fadel\OneDrive\Bureau\e-savior\SAVMAP_samples") 
    # main(dataset_dir)