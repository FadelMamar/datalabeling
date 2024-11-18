from datalabeling.arguments import Arguments, Dataprepconfigs
from datargs import parse
from pathlib import Path
import json
import logging


if __name__ == '__main__':

    args = parse(Arguments)
 
    if args.export_format is not None:
        from ultralytics import YOLO
        model = YOLO(args.export_model_weights)
        assert args.width==args.height,'Input image should have a square shape.'
        model.export(format=args.export_format,
                     imgsz=args.width,
                     nms=True,
                     half=args.half,
                     int8=args.int8,
                     batch=args.export_batch_size,
                     dynamic=args.dynamic,
                     simplify=True)
    
    if args.start_training:
        from datalabeling.train import start_training
        import mlflow
        import wandb
        import yaml

        if args.mlflow_model_alias is  None:
            logging.info(f"Loading model @ : {args.path_weights}")

        else:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            client = mlflow.MlflowClient()
            name = args.run_name
            alias = args.mlflow_model_alias
            version = client.get_model_version_by_alias(name=name,
                                                        alias=alias).version
            modelURI = f'models:/{name}/{version}'
            args.path_weights = mlflow.pyfunc.load_model(modelURI).unwrap_python_model().detection_model.model_path
            logging.info(f"Loading model registered with alias: {alias}")            

        with wandb.init(project=args.project_name,
                    config=args,
                    name=args.run_name,
                    tags=args.tag):
        
            # log data_config file
            with open(args.data_config_yaml,'r') as file:
                data_config = yaml.load(file,Loader=yaml.FullLoader)
                wandb.log(data_config)

            start_training(args)