#! /bin/bash


# CLEAR_ENABLED=$1
YAML=$1
SIZE=1024

DATASET_YAML=ultralytics/cfg/datasets/downtest.yaml
# DATASET_YAML=ultralytics/cfg/datasets/coco8.yaml

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
if [ ! $? -eq 0 ]; then
    echo "Failed to create directory ${SAVE_MODEL_PATH}"
    exit 1
fi

python3 train.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--yaml ${YAML}

if [ ! $? -eq 0 ]; then
    echo "Training failed"
    exit 1
fi

python3 export.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--model ${MODEL}

if [ ! $? -eq 0 ]; then
    echo "Export failed"
    exit 1
fi

edgetpu_compiler -s ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant.tflite
if [ ! $? -eq 0 ]; then
    echo "Edge TPU compilation failed"
    exit 1
fi

mv ${MODEL_NAME}.onnx ${MODEL_NAME}_full_integer_quant_edgetpu.log  ${MODEL_NAME}_full_integer_quant_edgetpu.tflite ${SAVE_MODEL_PATH}
mv runs/detect/train  ${SAVE_MODEL_PATH}
mv ${SAVE_MODEL_PATH}/train/weights/best.pt ${SAVE_MODEL_PATH}/${MODEL}
zip -r ${SAVE_MODEL_PATH}.zip ${SAVE_MODEL_PATH}
deactivate


source inference_venv/bin/activate
python3 inference.py \
--model ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
--data ${DATASET_YAML}

deactivate
