from ultralytics import YOLO, RTDETR
from ..arguments import Arguments
import os
from .utils import *



def training_routine(
    model: YOLO | RTDETR,
    args: Arguments,
    imgsz: int = None,
    batchsize: int = None,
    data_cfg: str | None = None,
    resume: bool = False,
):
    # Train the model
    model.train(
        data=data_cfg or args.data_config_yaml,
        epochs=args.epochs,
        imgsz=imgsz or min(args.height, args.width),
        device=args.device,
        freeze=args.freeze,
        name=args.run_name,
        single_cls=args.is_detector,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.optimizer_momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        batch=batchsize or args.batchsize,
        val=True,
        plots=True,
        cos_lr=args.cos_annealing,
        deterministic=False,
        cache=False,  # saves images as *.npy
        optimizer=args.optimizer,
        project=args.project_name,
        patience=args.patience,
        multi_scale=False,
        degrees=args.rotation_degree,
        mixup=args.mixup,
        scale=args.scale,
        mosaic=args.mosaic,
        augment=False,
        erasing=args.erasing,
        copy_paste=args.copy_paste,
        shear=args.shear,
        fliplr=args.fliplr,
        flipud=args.flipud,
        perspective=0.0,
        hsv_s=args.hsv_s,
        hsv_h=args.hsv_h,
        hsv_v=args.hsv_v,
        translate=args.translate,
        auto_augment="augmix",
        exist_ok=True,
        seed=args.seed,
        resume=resume,
    )


def pretraining_run(model: YOLO, args: Arguments):
    # check arguments
    assert os.path.exists(args.ptr_data_config_yaml), "provide --ptr-data-config-yaml"
    print("\n\n------------ Pretraining ----------", end="\n\n")
    # remove cache
    remove_label_cache(args.ptr_data_config_yaml)

    args.run_name = args.run_name + f"-PTR_freeze_{args.freeze}"

    # set parameters
    args.epochs = args.ptr_epochs
    args.lr0 = args.ptr_lr0
    args.lrf = args.ptr_lrf
    args.freeze = args.ptr_freeze
    training_routine(
        model=model,
        args=args,
        imgsz=args.ptr_tilesize,
        batchsize=args.ptr_batchsize,
        data_cfg=args.ptr_data_config_yaml,
        resume=False,
    )


def hard_negative_strategy_run(
    model: YOLO, args: Arguments, img_glob_pattern: str = "*"
):
    # check  arguments
    assert args.hn_save_dir is not None, "Provide --hn-save-dir"
    print(
        "\n\n------------ hard negative sampling learning strategy ----------",
        end="\n\n",
    )
    # remove cache
    remove_label_cache(args.hn_data_config_yaml)

    # update run_name
    args.run_name = args.run_name + f"-HN_freeze_{args.freeze}"

    cfg_path = get_data_cfg_paths_for_cl(
        ratio=args.hn_ratio,
        data_config_yaml=args.hn_data_config_yaml,
        cl_save_dir=args.hn_save_dir,
        seed=args.seed,
        split="train",
        pattern_glob=img_glob_pattern,
    )
    hn_cfg_path = get_data_cfg_paths_for_HN(args=args, data_config_yaml=cfg_path)
    args.lr0 = args.hn_lr0
    args.lrf = args.hn_lrf
    args.freeze = args.hn_freeze
    args.epochs = args.hn_num_epochs
    training_routine(
        model=model,
        args=args,
        imgsz=args.hn_imgsz,
        batchsize=args.hn_batch_size,
        data_cfg=hn_cfg_path,
        resume=False,
    )


def continual_learning_run(model: YOLO, args: Arguments, img_glob_pattern: str = "*"):
    # check arguments
    assert os.path.exists(args.cl_data_config_yaml), "Provide --cl-data-config-yaml"
    print("\n\n------------ Continual learning ----------", end="\n\n")
    # remove cache
    remove_label_cache(args.cl_data_config_yaml)
    # check flags
    for flag in (args.cl_ratios, args.cl_epochs, args.cl_freeze):
        assert len(flag) == len(args.cl_lr0s), (
            f"all args.cl_* flags should have the same length. {len(flag)} != {len(args.cl_lr0s)}"
        )
    
    # copy run_name
    run_name = args.run_name + ""

    # get yaml data_cfg files for CL runs
    count = 0
    for lr, ratio, num_epochs, freeze in zip(
        args.cl_lr0s, args.cl_ratios, args.cl_epochs, args.cl_freeze
    ):
        cl_cfg_path = get_data_cfg_paths_for_cl(
            ratio=ratio,
            data_config_yaml=args.cl_data_config_yaml,
            cl_save_dir=args.cl_save_dir,
            seed=args.seed,
            split="train",
            pattern_glob=img_glob_pattern,
        )
        # update run_name
        args.run_name = run_name + f"-CL_emptyRatio_{ratio}_freeze_{args.freeze}"
        # freeze layer. see ultralytics docs :)
        args.freeze = freeze
        args.lr0 = lr
        args.epochs = num_epochs
        training_routine(
            model=model,
            args=args,
            data_cfg=cl_cfg_path,
            batchsize=args.cl_batch_size,
            resume=False,
        )
        count += 1


def start_training(args: Arguments):
    """Trains a YOLO model using ultralytics.

    Args:
        args (Arguments): configs
    """

    # logger = logging.getLogger(__name__)
    assert args.task in ["detect", "obb", "segment"]

    # Load a pre-trained model
    model = YOLO(args.path_weights, task=args.task, verbose=False)
    if args.is_rtdetr:
        model = RTDETR(args.path_weights)

    # Display model information (optional)
    model.info()

    # pretraining
    if args.use_pretraining:
        pretraining_run(model=model, args=args)

    # Continual learning strategy
    if args.use_continual_learning:
        continual_learning_run(model=model, args=args)

    # hard negative sampling learning strategy
    if args.use_hn_learning:
        hard_negative_strategy_run(model=model, args=args)

    # standard training routine
    if not (
        args.ptr_data_config_yaml or args.use_continual_learning or args.use_hn_learning
    ):
        training_routine(model=model, args=args)
