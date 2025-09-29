#! /bin/bash
set -e
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
echo "AM AJUNS PANA AICI AM INSTALAT ELEMENTELE DE BAZA"

# First environment for training and exporting

curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile
source ~/.bashrc
source ~/.profile
pyenv install 3.11.1
pyenv shell 3.11.1
echo "AM INSTALAT PYENV CU SUCCES"
python3 -m venv venv
source venv/bin/activate
# pip install --upgrade pip
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip3 install -r requirements.txt
pip3 install roboflow
pip3 install dotenv
pip3 install tensorboard
yolo settings tensorboard=True
yolo settings datasets_dir=./datasets weights_dir=./weights runs_dir=./runs
echo "AM CREEAT ENVIRONMENT_UL"


mkdir datasets
cd utilitary
pip3 install gdown pandas pyyaml
python3 download.py
cp mydataset/dataset.yaml ../ultralytics/cfg/datasets
mv mydataset ../datasets
deactivate

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