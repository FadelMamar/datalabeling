import logging

from datargs import parse

from datalabeling.common.config import TrainingConfig
from datalabeling.common.pipeline import ModelTraining, Pipeline
from datalabeling.common.io import load_yaml

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    args = parse(TrainingConfig)

    # if args.export_format is not None:
    #     from ultralytics import YOLO

    #     model = YOLO(args.export_model_weights)
    #     assert args.width == args.height, "Input image should have a square shape."
    #     model.export(
    #         format=args.export_format,
    #         imgsz=args.width,
    #         nms=True,
    #         half=args.half,
    #         int8=args.int8,
    #         batch=args.export_batch_size,
    #         dynamic=args.dynamic,
    #         simplify=True,
    #     )

    # if args.start_training:
        # import wandb       

        # with wandb.init(
        #             # project=args.project_name,
        #             config=args,
        #             name=args.run_name,
        #             # tags=args.tag
        #             ):

        # log data_config file
        # if args.ptr_data_config_yaml:
        #     data_config = load_yaml(args.ptr_data_config_yaml)
        #     data_config["mode"] = "pretraining"
        #     # wandb.log(data_config)

        # if args.use_continual_learning:
        #     data_config = load_yaml(args.cl_data_config_yaml)
        #     data_config["mode"] = "continuous_learning"
        #     # wandb.log(data_config)

        # if args.use_hn_learning:
        #     data_config = load_yaml(args.hn_data_config_yaml)
        #     data_config["mode"] = "hard negative learning"
        #     # wandb.log(data_config)

        # if not (
        #     args.ptr_data_config_yaml
        #     or args.use_continual_learning
        #     or args.use_hn_learning
        # ):
        #     data_config = load_yaml(args.data_config_yaml)
        #     data_config["mode"] = "standard"
        #     wandb.log(data_config)



    # training_step = ModelTraining(
    #                 training_cfg=args,
    #                 herdnet_loss=None,
    #                 herdnet_training_backend=args.herdnet_training_backend,
    #                 model_type=args.model_type,
    # )

    # pipe = Pipeline(
    #     steps=[
    #         training_step,
    #     ]
    # )
    # pipe.run()


    # classification model
    from ultralytics import YOLO

    # Load a model
    model = YOLO(r"configs\yolo_configs\models\yolo12-cls.yaml").load('yolo11n-cls.pt')  # load a pretrained model (recommended for training)

    # model = YOLO("yolo11n-cls.pt")

    # # Train the model
    results = model.train(data=r"D:\PhD\Data per camp\Classification", 
                        epochs=50, 
                        imgsz=96,
                        batch=64,
                        project='classifier',
                        lrf=5e-3,
                        device='cuda:0',
                        cos_lr=True,
                        optimizer='auto',
                        auto_augment='augmix',
                        name='yolo12-cls'
                        )
