# Base image: NVIDIA CUDA 12.2 + cuDNN 8.9 + Ubuntu 22.04
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu20.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.kakao.com/ubuntu/|g' /etc/apt/sources.list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Install System Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-venv python3.8-dev python3-pip python3-tk \
    build-essential wget curl git unzip vim tmux nano ffmpeg \
    libgl1-mesa-glx libglib2.0-0 x11-apps libboost-all-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    tensorrt libnvinfer-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf python3.8 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Install Required Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# JAX with CUDA - 호환성 있는 버전으로 고정
RUN pip install --upgrade "jax[cuda12]" 
# -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Core packages
RUN pip install --no-cache-dir \
    dm-acme dm-env pygame gym opencv-python \
    flax optax==0.1.7 distrax chex rlax torch tensorflow \
    tensorboard matplotlib numpy pandas tqdm reeds-shepp 

# Jupyter
RUN pip install jupyterlab ipywidgets

# Other Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Setup Workspace
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
WORKDIR /mnt
COPY . .

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Set Default Command
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CMD ["bash"]