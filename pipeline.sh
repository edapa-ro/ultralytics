#! /bin/bash


# CLEAR_ENABLED=$1
YAML=$1
SIZE=1024

# DATASET_YAML=ultralytics/cfg/datasets/downtest.yaml
DATASET_YAML=ultralytics/cfg/datasets/coco8.yaml

MODEL_NAME=$(basename $YAML .yaml)
SAVE_MODEL_PATH=${MODEL_NAME}_saved_model
MODEL=${MODEL_NAME}.pt

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv shell 3.9.17

source venv/bin/activate
if [ ! -d ${SAVE_MODEL_PATH} ]
then
    mkdir ${SAVE_MODEL_PATH}
fi

python3 train.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--yaml ${YAML}

python3 export.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--model ${MODEL}
deactivate

source inference_venv/bin/activate

edgetpu_compiler -s ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant.tflite
mv ${MODEL_NAME}.onnx ${MODEL_NAME}_full_integer_quant_edgetpu.log  ${MODEL_NAME}_full_integer_quant_edgetpu.tflite ${SAVE_MODEL_PATH}

python3 inference.py \
--model ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
--data ${DATASET_YAML}
 
mv runs/detect/train  ${SAVE_MODEL_PATH}
mv ${SAVE_MODEL_PATH}/train/weights/best.pt ${SAVE_MODEL_PATH}/${MODEL}
