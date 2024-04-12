#!/bin/bash

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate label-backend

# source ./label-studio-env/bin/activate

label-studio-ml start HerdNet/my_ml_backend -p 9090

# deactivate