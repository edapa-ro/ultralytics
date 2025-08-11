#! /bin/bash

apt-get update
apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl zip
apt-get install -y curl gnupg ca-certificates apt-transport-https
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  > /etc/apt/sources.list.d/coral-edgetpu.list
apt-get update
apt-get install -y edgetpu-compiler


# First environment for training and exporting
set -e
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init --bash)"
exec "$SHELL"
pyenv install 3.9.17
pyenv shell 3.9.17
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install ultralytics
pip3 install tensorboard
yolo settings tensorboard=True
yolo settings datasets_dir=./datasets weights_dir=./weights runs_dir=./runs
deactivate

mkdir datasets
mv downtownwest2.zip datasets/
unzip datasets/downtownwest2.zip -d datasets/ 
mv datasets/my_dataset datasets/downtest

# Second environment for inference

# python3 -m venv inference_venv
# source inference_venv/bin/activate
# pip3 install numpy==1.26.4 pillow tensorflow pyyaml
# pip3 install tensorflow 
# pip3 install tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl
# pip3 install pycoral-2.0.0-cp39-cp39-linux_x86_64.whl

# pip3 install tensorboard
# yolo settings tensorboard=True
# yolo settings datasets_dir=./datasets weights_dir=./weights runs_dir=./runs


# utils/loss.py
# class DetectionModel(BaseModel): aici de inlocuit
# ultralytics/nn/tasks.py 501