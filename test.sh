#! /bin/bash

set -e
YAML=$1
SIZE=1024

DATASET_YAML=ultralytics/cfg/datasets/VisDrone.yaml
MODEL_NAME=$(basename $YAML .yaml)
SAVE_MODEL_PATH=${MODEL_NAME}_saved_model
MODEL=${MODEL_NAME}.pt

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv shell 3.11.1

source inference_venv/bin/activate
python3 inference.py \
--model ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
--data ${DATASET_YAML} \
--save "./res/${MODEL_NAME}"
deactivate

source venv/bin/activate
python3 visualize.py \
--data "./datasets/VisDrone/" \
--labels "./res/${MODEL_NAME}"
deactivate