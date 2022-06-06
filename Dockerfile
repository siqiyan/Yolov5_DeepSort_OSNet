ARG BASE_IMAGE
FROM $BASE_IMAGE

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV CUDA_HOME /usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# General dependencies:
ARG UPDATE_NV_KEYS
RUN if [ "$UPDATE_NV_KEYS" = true ] ; then \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub ; \
    fi

RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    gnupg2 \
    curl \
    lsb-core \
    vim \
    wget \
    apt-transport-https \
    ca-certificates\
    software-properties-common \
    git \
    unzip

RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
        ninja-build \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV (w/ Gstreamer backend) deps:
RUN apt-get update && apt-get install -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgtk2.0-dev \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libpython3-dev \
    python3-numpy

# Install miniconda to /miniconda
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init bash

# Setup Python environment
RUN conda create -n venv python=3.8 -y
RUN conda init bash
SHELL ["conda", "run", "--no-capture-output", "-n", "venv", "/bin/bash", "-c"]

# Build Pytorch
RUN conda install \
    astunparse \
    numpy \
    ninja \
    pyyaml \
    mkl \
    mkl-include \
    setuptools \
    cmake \
    cffi \
    typing_extensions \
    future \
    six \
    requests \
    dataclasses

RUN mkdir -p /deps

# Build Pytorch
RUN git clone --recursive https://github.com/pytorch/pytorch /deps/pytorch
RUN cd /deps/pytorch && \
    git checkout v1.8.0 && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0
RUN cd /deps/pytorch && python setup.py install

RUN git clone https://github.com/pytorch/vision.git /deps/torchvision
ENV FORCE_CUDA=1
RUN cd /deps/torchvision && \
    git checkout v0.9.0 && \
    python setup.py install

# Build OpenCV (w/ Gstreamer)
ENV CMAKE_ARGS="-DWITH_GSTREAMER=ON -DWITH_GTK=ON"
RUN git clone --recursive https://github.com/skvark/opencv-python.git /deps/opencv-python
RUN cd /deps/opencv-python && \
    pip install --upgrade pip wheel && \
    pip wheel . --verbose && \
    pip install opencv_python*.whl

ARG WORKDIR
RUN mkdir -p $WORKDIR
COPY . $WORKDIR/
WORKDIR $WORKDIR
RUN pip install -r requirements.txt

RUN echo "conda activate venv" >> /root/.bashrc
