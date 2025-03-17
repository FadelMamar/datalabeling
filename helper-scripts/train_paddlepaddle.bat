
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call .venv-paddle\Scripts\activate.bat

call uv run tools\paddlepaddle.py --do-eval True --config data\ppd_configs\faster_rcnn_r50_fpn_1x_coco.yml ^
    --weights base_models_weights\ResNet50_cos_pretrained.pdparams ^
    --run-name "faster_rcnn_r50_fpn" --output-dir ppd_runs/faster_rcnn_r50_fpn