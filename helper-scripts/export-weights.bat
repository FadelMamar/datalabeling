call .\activate_label-backend_env.bat

call cd ..

call python tools\cli.py --export-format "openvino" --width 640 --height 640 --export-batch-size 1^
             --export-model-weights "C:\Users\FADELCO\OneDrive\Bureau\datalabeling\models\best.pt"^
             --half

call conda deactivate