# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################
# SDK Build Image, used to copy installs
############################################################
ARG BUILD_IMAGE
FROM ${BUILD_IMAGE} AS build

############################################################
# Base Image
############################################################
# CUDA-base image is ~250 MB and contains the CUDA runtime + sets PATH and LD_LIBRARY_PATH
FROM nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04 AS base

# Variables inherited by all downstream stages
ARG HOST_INSTALL_DIR
ARG MAX_PROC=32
ARG INSTALL_PATH=/opt/nvidia/holoscan
ARG GPU_TYPE=dgpu
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

############################################################
# Runtime C++ (no MKL)
#
# This stage includes all that is needed to run C++ Holoscan
# apps on ARM64 platforms and x86_64, apart from MKL that is
# needed on x86_64 only for libtorch backend.
############################################################
FROM base AS runtime_cpp_no_mkl

# Install apt run dependencies
#  libx* - X packages
#  libvulkan1 - for Vulkan apps (Holoviz)
#  libegl1 - to run headless Vulkan apps
#  libwayland-client0, libwayland-egl1, libxkbcommon0 - GLFW runtime dependency for Wayland
#  libdecor-0-plugin-1-cairo - GLFW runtime dependency for Wayland window decorations
#  libopenblas0 - libtorch dependency
#  libnuma1 - libtorch dependency
#  libgomp1 - libtorch & CuPy dependency
#  libv4l2 - V4L2 operator dependency
#  libpng16-16 - torchvision dependency
#  libjpeg8 - torchvision dependency
#  libnvonnxparsers - TensorRT dependency
#  libnccl2 - libtorch & CuPy dependency
#  libnvinfer-plugin - TensorRT dependency (Also installs libnvinfer8, libcublas, and libcudnn)
#  libcublas - TensorRT & OnnxRT dependency, installed by libnvinfer8 (Installed explicitly to ensure correct version)
#  libnpp - Needed by format_converter & bayer_demosaic
#  libcufft - Holoscan-python-core/OnnxRT dependency
#  libcurand - libtorch & CuPy dependency
#  libcusparse - libtorch & CuPy dependency
#  cuda-nvrtc - libtorch & CuPy dependency
#  libnvjitlink - libtorch & CuPy dependency
#  libcusolver - libtorch & CuPy dependency
#  cuda-cupti - libtorch & CuPy dependency
#  cuda-nvtx - libtorch & CuPy dependency
#  libcudnn_train.so is removed since training is not needed in a runtime environment (saves ~200 MB)
RUN apt-get update \
    && apt-get install --no-install-recommends --allow-downgrades -y \
        libx11-6="2:1.7.5-*" \
        libxcursor1="1:1.2.0-*" \
        libxi6="2:1.8-*" \
        libxinerama1="2:1.1.4-*" \
        libxrandr2="2:1.5.2-*" \
        libvulkan1="1.3.204.1-*" \
        libegl1="1.4.0-*" \
        libwayland-client0="1.20.0-*" \
        libwayland-egl1="1.20.0-*" \
        libxkbcommon0="1.4.0-*" \
        libdecor-0-plugin-1-cairo="0.1.0-*" \
        libopenblas0="0.3.20+ds-*" \
        libnuma1="2.0.14-*" \
        libgomp1="12.3.0-*" \
        libv4l-0="1.22.1-*" \
        libpng16-16="1.6.37-*" \
        libjpeg-turbo8="2.1.2-*" \
        libnvinfer-plugin8="8.6.*+cuda12.0" \
        libnvonnxparsers8="8.6.*+cuda12.0" \
        libnccl2="2.19*+cuda12.2" \
        libcublas-12-2 \
        libnpp-12-2 \
        libcufft-12-2 \
        libcurand-12-2 \
        libcusparse-12-2 \
        cuda-nvrtc-12-2 \
        libnvjitlink-12-2 \
        libcusolver-12-2 \
        cuda-cupti-12-2 \
        cuda-nvtx-12-2 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/*/libcudnn*train.so*

# Copy ONNX Runtime
ARG ONNX_RUNTIME_VERSION=1.15.1_23.08
ENV ONNX_RUNTIME=/opt/onnxruntime/${ONNX_RUNTIME_VERSION}/lib
COPY --from=build ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${ONNX_RUNTIME}"

# Copy Libtorch
ARG LIBTORCH_VERSION=2.1.0_23.08
ENV LIBTORCH=/opt/libtorch/${LIBTORCH_VERSION}/lib
COPY --from=build ${LIBTORCH} ${LIBTORCH}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${LIBTORCH}"

# Copy TorchVision
ARG TORCHVISION_VERSION=0.16.0_23.08
ENV TORCHVISION=/opt/torchvision/${TORCHVISION_VERSION}/lib
COPY --from=build ${TORCHVISION} ${TORCHVISION}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${TORCHVISION}"

# Copy inherited Torch library dependency that result from building with NGC PyTorch container
COPY --from=build /opt/hpcx/ucc/lib/libucc.so.1 $INSTALL_PATH/lib/libucc.so.1
COPY --from=build /opt/hpcx/ompi/lib/libmpi.so.40 $INSTALL_PATH/lib/libmpi.so.40
COPY --from=build /opt/hpcx/ompi/lib/libopen-rte.so.40 $INSTALL_PATH/lib/libopen-rte.so.40
COPY --from=build /opt/hpcx/ompi/lib/libopen-pal.so.40 $INSTALL_PATH/lib/libopen-pal.so.40

# Install GRDAPI (needed by Holoscan-core)
ENV os=ubuntu2204
ENV tag=8.6.1-cuda-12.0
RUN if [ $(uname -m) = "aarch64" ]; then \
        GDR_REPO_ARCH=aarch64 DEB_ARCH=arm64 CUDA_REPO_ARCH=arm64 ; \
    else \
        GDR_REPO_ARCH=x64 DEB_ARCH=amd64 CUDA_REPO_ARCH=x86_64 ; \
    fi \
    && apt-get update -y \
    && apt-get install --no-install-recommends -y curl \
    && curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${CUDA_REPO_ARCH}/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && curl -O https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.2/ubuntu22_04/${GDR_REPO_ARCH}/libgdrapi_2.4-1_${DEB_ARCH}.Ubuntu22_04.deb \
    && dpkg -i libgdrapi_2.4-1_${DEB_ARCH}.Ubuntu22_04.deb \
    && rm -f libgdrapi_2.4-1_${DEB_ARCH}.Ubuntu22_04.deb \
    && apt-get remove --purge -y curl \
    && rm -rf /var/lib/apt/lists/*

## SDK Runtime Install
# Only /bin, /lib, and /python components are needed
COPY ${HOST_INSTALL_DIR}/bin $INSTALL_PATH/bin
COPY ${HOST_INSTALL_DIR}/lib $INSTALL_PATH/lib
COPY ${HOST_INSTALL_DIR}/python $INSTALL_PATH/python
ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${INSTALL_PATH}
ENV HOLOSCAN_LIB_PATH=${INSTALL_PATH}/lib
ENV PYTHONPATH=${INSTALL_PATH}/python/lib
ENV PATH=${PATH}:${INSTALL_PATH}/bin
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOLOSCAN_LIB_PATH}"

# Start in the install path
WORKDIR $INSTALL_PATH

#############################################################
# Runtime C++ + pip + MKL
#
# MKL is an x8_64 C++ Holoscan dependency installed with PIP,
# so Python must be installed. However, this stage does NOT
# support running Python Holoscan apps.
#############################################################
FROM runtime_cpp_no_mkl AS runtime_cpp_pip_mkl

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        gnupg \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install --no-install-recommends -y \
        python3-pip \
    && add-apt-repository --remove ppa:deadsnakes/ppa \
    && apt-get remove --purge -y \
        gnupg \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

#  mkl - dependency for libtorch plugin on x86_64 (match pytorch container version)
RUN if [ $(uname -m) = "x86_64" ]; then \
        python3 -m pip install --no-cache-dir \
            mkl==2021.1.1 \
        && \
        # Clean up duplicate libraries from mkl/tbb python wheel install which makes copies for symlinks.
        # Only keep the *.so.X libs, remove the *.so and *.so.X.Y libs
        # This can be removed once upgrading to an MKL pip wheel that fixes the symlinks
        find /usr/local/lib -maxdepth 1 -type f -regex '.*\/lib\(tbb\|mkl\).*\.so\(\.[0-9]+\.[0-9]+\)?' -exec rm -v {} +; \
    fi

#############################################################
# Runtime C++
#############################################################
FROM --platform=linux/arm64 runtime_cpp_no_mkl AS runtime_cpp_arm64
FROM --platform=linux/amd64 runtime_cpp_pip_mkl AS runtime_cpp_amd64
FROM runtime_cpp_${TARGETARCH} AS runtime_cpp

############################################################
# Runtime (C++ & Python)
#
# This stage adds Python dependencies to support both C++
# and Python Holoscan applications.
############################################################
FROM runtime_cpp_pip_mkl AS runtime_cpp_py

# Install pip run dependencies
#  requirements
#    pip - 20.3+ needed for PEP 600
#    cupy-cuda - dependency for holoscan python + examples
#    cloudpickle - dependency for distributed apps
#    python-on-whales - dependency for holoscan CLI
#    Jinja2 - dependency for holoscan CLI
#    packaging - dependency for holoscan CLI
#    pyyaml - dependency for holoscan CLI
#    requests - dependency for holoscan CLI
#    psutil - dependency for holoscan CLI
COPY python/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
