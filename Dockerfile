FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Base dependencies
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get -y install build-essential cmake git doxygen graphviz pkg-config libeigen3-dev wget

WORKDIR /root

# Eigen
RUN git clone https://gitlab.com/libeigen/eigen.git && \
  cd eigen && \
  git checkout 27367017bd0aef15a67ce76b8e263a94c2508a1c && \
  cmake -S . -B build -DCMAKE_INSTALL_PREFIX=~/.local && \
  cmake --build build && \
  cmake --build build --target install

# OpenCV
RUN git clone https://github.com/opencv/opencv.git && \
  cd opencv && \
  git checkout 82ac7ea23620fb13b7b6be225fa1b0e848f5e72d
RUN git clone https://github.com/opencv/opencv_contrib.git && \
  cd opencv_contrib && \
  git checkout c4027ab7f912a3053175477d41b1a62d0078bc5f
RUN cd opencv && \
  cmake -S . -B build -DOPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -DCMAKE_INSTALL_PREFIX=~/.local && \
  cmake --build build && \
  cmake --build build --target install

# ETTCM
RUN mkdir ETTCM
COPY . ETTCM

WORKDIR /root/ETTCM

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR=~/.local/share/eigen3/cmake -DOpenCV_DIR=~/.local/lib/cmake/opencv4 && \
  cmake --build build && \
  make -C build doc
