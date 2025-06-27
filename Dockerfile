# Base image: NVIDIA CUDA 11.8 + cuDNN 8 + Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Install System Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-venv python3.8-dev python3-pip python3-tk \
    build-essential wget curl git unzip vim tmux nano ffmpeg \
    libgl1-mesa-glx libglib2.0-0 x11-apps libboost-all-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf python3.8 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Install Required Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Jax with CUDA
RUN pip install --upgrade "jax[cuda11_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Core packages
RUN pip install --no-cache-dir \
    dm-acme dm-env pygame gym opencv-python \
    chex flax rlax torch tensorflow==2.11 \
    "optax<0.1.7" tensorboardX \
    matplotlib numpy pandas tqdm reeds-shepp 

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
