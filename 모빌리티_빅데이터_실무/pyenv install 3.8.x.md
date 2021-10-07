pyenv install 3.8.x

pyenv virtualenv 3.8.x py38dl
mkdir -p ~/workspace/detlec
cd ~/workspace/detlec
pyenv local py38dl
pip install --upgrade pip

pip install numpy opencv-python scikit-learn matplotlib pydot
sudo apt install graphviz

pip install tensorflow==2.4

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html