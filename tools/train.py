from ultralytics import YOLO
from datargs import parse
from arguments import Arguments
import os
import yaml
import wandb
from datalabeling.train import start_training


if __name__ == '__main__':
    args = parse(Arguments)

    # if __name__ == '__main__':
#     args = parse(Arguments)
#     start_training(args=args)

    with wandb.init(project=args.project_name,
                     config=args,
                    name=args.run_name,
                    tags=args.tag):
        
        # log data_config file
        with open(args.data_config_yaml,'r') as file:
            data_config = yaml.load(file,Loader=yaml.FullLoader)
            wandb.log(data_config)

        start_training(args=args)





