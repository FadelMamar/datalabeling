call .\activate_label-backend_env.bat

call cd ..

call python tools\cli.py --export-format "openvino" --width 1280 --height 1280 --export-batch-size 1^
             --export-model-weights "C:\Users\FADELCO\OneDrive\Bureau\datalabeling\models\best.pt"^
             --half --dynamic

call conda deactivate