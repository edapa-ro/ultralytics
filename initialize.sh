#! /bin/bash
apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl zip
curl https://pyenv.run | bash
echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
pyenv install 3.9.17
pyenv shell 3.9.17
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install ultralytics
pip3 install tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl
pip3 install pycoral-2.0.0-cp39-cp39-linux_x86_64.whl
yolo settings tensorboard=True
yolo settings datasets_dir=./datasets weights_dir=./weights runs_dir=./runs


# utils/loss.py
# class DetectionModel(BaseModel): aici de inlocuit
# ultralytics/nn/tasks.py 501