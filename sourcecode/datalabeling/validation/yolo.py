from ultralytics import YOLO, RTDETR


def ultralytics_validate(path, data_config_yaml, split, imgsz, conf_threshold, iou_threshold,model_type='yolo', device="cuda", batch=16):

    if model_type == 'yolo':
        model = YOLO(path)
    elif model_type == 'rtdetr':
        model = RTDETR(path)
    else:
        raise NotImplementedError(f"{model_type} is not supproted.")

    model.info()
    
    # Customize validation settings
    validation_results = model.val(data=data_config_yaml,
                                    imgsz=imgsz,
                                    batch=batch,
                                    split=split,
                                    conf=conf_threshold,
                                    iou=iou_threshold,
                                    device=device
                                )
    return validation_results