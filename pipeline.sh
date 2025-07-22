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

pip3 install opencv-python
python3 train.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--yaml ${YAML}
# --model ${MODEL}

python3 export.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--model ${MODEL}

edgetpu_compiler ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant.tflite
cat ${MODEL_NAME}_full_integer_quant_edgetpu.log
mv ${MODEL_NAME}.onnx ${MODEL_NAME}_full_integer_quant_edgetpu.log  ${MODEL_NAME}_full_integer_quant_edgetpu.tflite ${SAVE_MODEL_PATH}

pip3 uninstall opencv-python -y
pip3 install numpy==1.26.4
python3 inference.py \
--model ${SAVE_MODEL_PATH}/${MODEL_NAME}_full_integer_quant_edgetpu.tflite \
--data ${DATASET_YAML}
 
mv runs/detect/train  ${SAVE_MODEL_PATH}
mv runs/detect/train/weights/best.pt ${SAVE_MODEL_PATH}/${MODEL}
tensorboard --logdir ${SAVE_MODEL_PATH}/train
