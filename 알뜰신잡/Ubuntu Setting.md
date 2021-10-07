Ubuntu Setting
===

- Ubuntu 20.04 Download: https://ubuntu.com/download/desktop

- 우분투 기본 설치

  - `sudo apt update && sudo apt upgrade -y`
  - `sudo apt install build-essential cmake git curl`

  

- NVIDIA driver: `sudo ubuntu-drivers autoinstall`

  - ubuntu-drivers devices 
    - 현재 그래픽 카드에 적합한 드라이버를 보여준다.
  - nvidia-smi
    - 설치된 드라이버의 정보를 보여준다.

```shell
$ sudo ubuntu-drivers devices
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ apt-cache search nvidia | grep nvidia-driver-450
$ sudo apt-get install nvidia-driver-450
$ S

$ sudo nvidia-smi
or

$ sudo ubuntu-drivers autoinstall


---
sudo apt-get remove --purge '^nvidia-.*' 
sudo apt-get --purge remove 'cuda*'
sudo apt-get autoremove --purge 'cuda*'

sudo rm -rf /usr/local/cuda*
```



---

- CUDA Download: [https://developer.nvidia.com/cuda-toolkit-archiv](https://developer.nvidia.com/cuda-toolkit-archive)

```shell
$ wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
$ sudo sh cuda_11.0.3_450.51.06_linux.run
# sudo sh cuda_11.0.2_450.51.05_linux.run

or
$ chmod a+x cuda_<version>.run
$ sudo ./cuda_<version>.run

> 그래픽부분만 체크해제하고 시작하면 된다.
---

$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

$ wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb

$ sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

그래픽을 설치하면 자동으로 CUDA가 설치되는 부분이 있음.

run파일로 실행에서 deb 파일 설치하여 local에 cuda폴더를 만들기로 함.





``` python
$ sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.2/bin' >> /etc/profile"
$ sudo sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64' >> /etc/profile"
$ sudo sh -c "echo 'export CUDADIR=/usr/local/cuda-11.2' >> /etc/profile"
$ source /etc/profile

nvcc -V
```



---

- cuDNN: https://developer.nvidia.com/rdp/cudnn-archive
  - NVIDIA 계정 로그인
  - Select “cuDNN-xxx for CUDA 11.0”
  - Download “cuDNN Library for Linux (x86_64)”
  - tgz 압축해제
  - 헤더와 라이브러리 복사

```
sudo cp <extracted directory>/cuda/include/* /usr/local/cuda/include
sudo cp <extracted directory>/cuda/lib64/* /usr/local/cuda/lib64


sudo cp cuda/include/* /usr/local/cuda/include
sudo cp cuda/lib64/* /usr/local/cuda/lib64

$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

​	

- `~/.profile`에 아래 내용 없으면 추가

```
export LD_LIBARARY_PATH="/usr/local/cuda/lib64:${LD_LIBARARY_PATH}"
export PATH="/usr/local/cuda/bin:${PATH}"
```



```python
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.0  /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.0  /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.0  /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.0  /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
$ sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8.1.0  /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8

ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn


```





```
sudo ln -s /usr/local/cuda/lib64/libcusolver.so.11 ~/.pyenv/versions/py388ver/lib/python3.8/site-packages/tensorflow/python/libcusolver.so.10

```



---



- Pyenv: https://github.com/pyenv/pyenv-installer

  - pyenv prerequisites: `https://github.com/pyenv/pyenv/wiki/Common-build-problems`

  - ```shell
    $ sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
    ```

    이거 꼭해라..




  - `curl https://pyenv.run | bash`

  - add the three lines to `~/.bashrc`

    ```
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    # 만약 에러날 시 
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
    ```

---

  PyCharm: `sudo snap install pycharm-community --classic`

- ``` python
  # Ctrl+Alt+T 로 터미널 열기
  # 압축을 풀어 /opt 경로에 놓기
  sudo tar -xf /home/pi/Downloads/pycharm-community-2020.2.1.tar.gz -C /opt
  # 사용사 설정 파일 열기
  mousepad ~/.bashrc
  # 파이참 실행 명령어 등록 위해 맨 아래 줄에 한 줄 추가
  alias pycharm="/opt/pycharm-community-2020.2.1/bin/pycharm.sh"
  # Ctrl+S 눌러 저장후 오른쪽 위 'x' 눌러 닫기
  
  # 터미널에서 파이참 실행
  pycharm
  ```

  

---

- 한글 설치
  - Settings -> Region & Language -> Add “Korean(101/104 key compatible)” -> Manage Installed Languages -> Update
  - `sudo apt install fcitx-hangul`
  - Settings -> Region & Language -> Manage Installed Languages -> Keyboard input method system: “fcitx”
  - 재부팅
  - 한/영키 등록: 오른쪽 위 키보드 아이콘 -> Configure -> “+” -> “Hangul” 추가 -> Global Config 탭 -> Trigger Input Method
- Naver Whale: https://whale.naver.com/ko/download

---



Pyenv Setting
===

Pyenv를 이용한 파이썬 설치

`pyenv install 3.8.x`(latest)

`pyenv install 3.8.7`(로 설치하여 진행하였음)



Setup virtual Environment
---

```shell
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
```



Test TF2 and Pytorch

- Open PtCharm at `~/workspace/detlec`

- Set pycharm interpreter at `~/.pyenv/versions/py38dl/bin/python`

- Test tensorflow by

  ```python
  python 
  import tensorflow as tf 
  x = tf.random.uniform((2,3,4)) 
  print(tf.reduce_sum(x, axis=(1,2)))
  ```

---

Output : tf.Tensor([4.5362177 5.611371], shape=(2,), dtype=float32)

```python
- Test pytorch by

  ```python
  import tensorflow as tf
  x = tf.random.uniform((2, 3, 4))
  print(tf.reduce_sum(x, axis=(1, 2)))
  
  ---------------------------
  Output:
  tensor([8.4671, 4.6608])
```

---

~/workspace/detlec/venv/bin/python

`pyenv install --list | grep 3.8` 3.8버전 리스트만 보여준다.

 tf.config.list_physical_devices('GPU')

///https://webnautes.tistory.com/1428





++

460 - 11.2 / 8.1 / tensor 2.5 사용하기

450 - 11.0 / 8.0 / tensor 2.4 



cuda 랑 tensorflow랑 버전 안맞으면 에러나고 gpu 할당 안됨...