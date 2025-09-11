#! /bin/bash

# https://github.com/google-coral/webcoral.git
set -e
YAML=$1
SIZE=1024

# DATASET_YAML=ultralytics/cfg/datasets/VisDrone.yaml
# DATASET_YAML=ultralytics/cfg/datasets/downtest.yaml
DATASET_YAML=ultralytics/cfg/datasets/coco.yaml

MODEL_NAME=$(basename $YAML .yaml)
SAVE_MODEL_PATH=${MODEL_NAME}_saved_model
MODEL=${MODEL_NAME}.pt

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv shell 3.11.1

source venv/bin/activate
if [ ! -d ${SAVE_MODEL_PATH} ]
then
    mkdir ${SAVE_MODEL_PATH}
fi

python3 resize_visdrone.py \
--size ${SIZE} \
--yaml ${DATASET_YAML} 

python3 train.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--yaml ${YAML} \
--model ${MODEL} 

cp runs/detect/train/weights/best.pt ${MODEL}
python3 export.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--model ${MODEL}
deactivate

edgetpu_compiler -sa ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant.tflite
mv ${MODEL_NAME}.onnx ${MODEL_NAME}_full_integer_quant_edgetpu.log  ${MODEL_NAME}_full_integer_quant_edgetpu.tflite ${SAVE_MODEL_PATH}

mv runs/detect/train  ${SAVE_MODEL_PATH}
rm -r ${SAVE_MODEL_PATH}/saved_model.pb ${SAVE_MODEL_PATH}/fingerprint.pb ${SAVE_MODEL_PATH}/assets ${SAVE_MODEL_PATH}/variables ${SAVE_MODEL_PATH}/metadata.yaml
rm -r ${SAVE_MODEL_PATH}/train/*.png ${SAVE_MODEL_PATH}/train/*.jpg
rm -r ${SAVE_MODEL_PATH}/${MODEL_NAME}_float16.tflite ${SAVE_MODEL_PATH}/${MODEL_NAME}_float32.tflite ${SAVE_MODEL_PATH}/${MODEL_NAME}.onnx
rm -r ${SAVE_MODEL_PATH}/train/weights/epoch*.pt
mv ${SAVE_MODEL_PATH}/train/weights/best.pt ${SAVE_MODEL_PATH}/${MODEL}
zip -r archive_${SAVE_MODEL_PATH}.zip ${SAVE_MODEL_PATH}

source inference_venv/bin/activate
python3 inference.py \
--model ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
--data ${DATASET_YAML} \
--save "./res/${MODEL_NAME}"
deactivate

source venv/bin/activate
python3 visualize.py \
--data "./datasets/downtest/" \
--labels "./res/${MODEL_NAME}"
deactivate
