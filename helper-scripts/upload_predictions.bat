call .\activate_label-backend_env.bat

call cd ..\tools

@REM arguments
@REM project_id:int,
@REM path_to_weights: str,
@REM is_yolo_obb:bool,
@REM tilesize:int, 
@REM overlapratio:float,
@REM confidence_threshold:float,
@REM device:str
@REM use_sliding_window
call uv run upload_predictions.py 1 "..\models\best_openvino_model" True 1280 0.1 0.2 "NPU" False

call deactivate