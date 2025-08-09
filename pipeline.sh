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
deactivate

mv runs/detect/train  ${SAVE_MODEL_PATH}
rm -r ${SAVE_MODEL_PATH}/save_model.pb ${SAVE_MODEL_PATH}/fingerprint.pb ${SAVE_MODEL_PATH}/assets ${SAVE_MODEL_PATH}/variables ${SAVE_MODEL_PATH}/metada.yaml
rm -r ${SAVE_MODEL_PATH}/train/*.png ${SAVE_MODEL_PATH}/train/*.jpg
rm -r ${SAVE_MODEL_PATH}/half_channels_yolov5C2fGhost_float16.tflite ${SAVE_MODEL_PATH}/half_channels_yolov5C2fGhost_float32.tflite ${SAVE_MODEL_PATH}/half_channels_yolov5C2fGhost.onnx
rm -r ${SAVE_MODEL_PATH}/train/weights/epoch*.pt
mv ${SAVE_MODEL_PATH}/train/weights/best.pt ${SAVE_MODEL_PATH}/${MODEL}
zip -r archive_${SAVE_MODEL_PATH}.zip ${SAVE_MODEL_PATH}


source inference_venv/bin/activate
python3 inference.py \
--model ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
--data ${DATASET_YAML}
deactivate

if [ ! $? -eq 0 ]; then
    echo "Inference failed"
    exit 1
fi

source venv/bin/activate
python3 visualize.py \
--data "./datasets/downtest/" \
--labels "./res/labels/test/"
deactivate
