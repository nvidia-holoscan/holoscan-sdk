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
# Versions
############################################################
# Dependencies ending in _YY.MM are built or extracted from
# the TensorRT or PyTorch NGC containers of that same version
ARG ONNX_RUNTIME_VERSION=1.18.1_38712740_24.08-cuda-12.6
ARG LIBTORCH_VERSION=2.5.0_24.08
ARG TORCHVISION_VERSION=0.20.0_24.08
ARG GRPC_VERSION=1.54.2
ARG GXF_VERSION=4.1.1.4_20241210_dc72072
ARG MOFED_VERSION=24.07-0.6.1.0

############################################################
# Base image
############################################################
ARG GPU_TYPE=dgpu
FROM nvcr.io/nvidia/tensorrt:24.08-py3 AS dgpu_base
FROM nvcr.io/nvidia/tensorrt:24.08-py3-igpu AS igpu_base
FROM ${GPU_TYPE}_base AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

############################################################
# Variables
############################################################
ARG MAX_PROC=32

############################################################
# Build tools
############################################################
FROM base AS build-tools

# Install build tools
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        patchelf \
        ninja-build="1.10.1-*" \
    && rm -rf /var/lib/apt/lists/*

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

############################################################
# NGC CLI
############################################################
FROM base AS ngc-cli-downloader

WORKDIR /opt/ngc-cli

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && curl -S -# -L -o ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -q ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && export ngc_exec=$(find . -type f -executable -name "ngc" | head -n1)

############################################################
# ONNX Runtime
############################################################
FROM base AS onnxruntime-downloader
ARG ONNX_RUNTIME_VERSION

# Download ORT binaries from artifactory
# note: built with CUDA and TensorRT providers
WORKDIR /opt/onnxruntime
RUN curl -S -L -# -o ort.tgz \
    https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/onnxruntime/onnxruntime-${ONNX_RUNTIME_VERSION}-$(uname -m).tar.gz
RUN mkdir -p ${ONNX_RUNTIME_VERSION}
RUN tar -xf ort.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 2

############################################################
# Libtorch
############################################################
FROM base AS torch-downloader
ARG LIBTORCH_VERSION
ARG GPU_TYPE

# Download libtorch binaries from artifactory
# note: extracted from nvcr.io/nvidia/pytorch:24.08-py3
WORKDIR /opt/libtorch/
RUN ARCH=$(uname -m) && if [ "$ARCH" = "aarch64" ]; then ARCH="${ARCH}-${GPU_TYPE}"; fi && \
    curl -S -# -o libtorch.tgz -L \
        https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/libtorch/libtorch-${LIBTORCH_VERSION}-${ARCH}.tar.gz
RUN mkdir -p ${LIBTORCH_VERSION}
RUN tar -xf libtorch.tgz -C ${LIBTORCH_VERSION} --strip-components 1

# Patch step to remove kineto from config to remove warning, not needed by holoscan
RUN find . -type f -name "*Config.cmake" -exec sed -i '/kineto/d' {} +
# Patch step for CMake configuration warning
COPY patches/libtorch.Caffe2.cmake.patch ${LIBTORCH_VERSION}/share/cmake/Caffe2/cuda.patch
WORKDIR ${LIBTORCH_VERSION}
RUN patch -p1 < share/cmake/Caffe2/cuda.patch

############################################################
# TorchVision
############################################################
FROM base AS torchvision-downloader
ARG TORCHVISION_VERSION
ARG GPU_TYPE

# Download torchvision binaries from artifactory
# note: extracted from nvcr.io/nvidia/pytorch:23.07-py3
WORKDIR /opt/torchvision/
RUN ARCH=$(uname -m) && if [ "$ARCH" = "aarch64" ]; then ARCH="${ARCH}-${GPU_TYPE}"; fi && \
    curl -S -# -o torchvision.tgz -L \
        https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/torchvision/torchvision-${TORCHVISION_VERSION}-${ARCH}.tar.gz
RUN mkdir -p ${TORCHVISION_VERSION}
RUN tar -xf torchvision.tgz -C ${TORCHVISION_VERSION} --strip-components 1

############################################################
# gRPC libraries and binaries
############################################################
FROM build-tools AS grpc-builder
ARG GRPC_VERSION

WORKDIR /opt/grpc
RUN git clone --depth 1 --branch v${GRPC_VERSION} \
         --recurse-submodules --shallow-submodules \
         https://github.com/grpc/grpc.git src
RUN cmake -S src -B build -G Ninja \
         -D CMAKE_BUILD_TYPE=Release \
         -D CMAKE_CXX_VISIBILITY_PRESET=hidden \
         -D CMAKE_VISIBILITY_INLINES_HIDDEN=1 \
         -D gRPC_INSTALL=ON \
         -D gRPC_BUILD_TESTS=OFF
RUN cmake --build build -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN cmake --install build --prefix /opt/grpc/${GRPC_VERSION}

############################################################
# MOFED
############################################################
FROM build-tools AS mofed-installer
ARG MOFED_VERSION

# In a container, we only need to install the user space libraries, though the drivers are still
# needed on the host.
# Note: MOFED's installation is not easily portable, so we can't copy the output of this stage
# to our final stage, but must inherit from it. For that reason, we keep track of the build/install
# only dependencies in the `MOFED_DEPS` variable (parsing the output of `--check-deps-only`) to
# remove them in that same layer, to ensure they are not propagated in the final image.
WORKDIR /opt/nvidia/mofed
ARG MOFED_INSTALL_FLAGS="--dpdk --with-mft --user-space-only --force --without-fw-update"
RUN UBUNTU_VERSION=$(cat /etc/lsb-release | grep DISTRIB_RELEASE | cut -d= -f2) \
    && OFED_PACKAGE="MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu${UBUNTU_VERSION}-$(uname -m)" \
    && curl -S -# -o ${OFED_PACKAGE}.tgz -L \
        https://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/${OFED_PACKAGE}.tgz \
    && tar xf ${OFED_PACKAGE}.tgz \
    && MOFED_INSTALLER=$(find . -name mlnxofedinstall -type f -executable -print) \
    && MOFED_DEPS=$(${MOFED_INSTALLER} ${MOFED_INSTALL_FLAGS} --check-deps-only 2>/dev/null | tail -n1 |  cut -d' ' -f3-) \
    && apt-get update \
    && apt-get install --no-install-recommends -y ${MOFED_DEPS} \
    && ${MOFED_INSTALLER} ${MOFED_INSTALL_FLAGS} \
    && rm -r * \
    && apt-get remove -y ${MOFED_DEPS} && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

############################################################
# UCX
############################################################
FROM build-tools AS ucx-patcher

# The base container provides custom builds of HPCX libraries without
# the necessary rpath for non-containerized applications. We patch RPATH
# for portability when we later repackage these libraries for distribution
# outside of the container.
RUN patchelf --set-rpath '$ORIGIN' /opt/hpcx/ucx/lib/libuc*.so* \
    && patchelf --set-rpath '$ORIGIN:$ORIGIN/..' /opt/hpcx/ucx/lib/ucx/libuc*.so* \
    && patchelf --set-rpath '$ORIGIN/../lib' /opt/hpcx/ucx/bin/*

############################################################
# GXF
############################################################
FROM base AS gxf-downloader
ARG GXF_VERSION

WORKDIR /opt/nvidia/gxf
RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=x86_64; fi \
    && curl -S -# -L -o gxf.tgz \
        https://urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/gxf/gxf_${GXF_VERSION}_holoscan-sdk_${ARCH}.tar.gz
RUN mkdir -p ${GXF_VERSION}
RUN tar xzf gxf.tgz -C ${GXF_VERSION} --strip-components 1

############################################################
# Build image (final)
############################################################
FROM mofed-installer AS build

## COPY
# Note: avoid LD_LIBRARY_PATH - https://www.hpc.dtu.dk/?page_id=1180

# Copy NGC CLI
ENV NGC_CLI=/opt/ngc-cli
COPY --from=ngc-cli-downloader ${NGC_CLI}/ngc-cli ${NGC_CLI}
ENV PATH="${PATH}:${NGC_CLI}"
# Workaround to fix encoding for ngc-cli download
# https://jirasw.nvidia.com/browse/NGC-31306
ENV PYTHONIOENCODING=utf-8 LC_ALL=C.UTF-8

# Copy ONNX Runtime
ARG ONNX_RUNTIME_VERSION
ENV ONNX_RUNTIME=/opt/onnxruntime/${ONNX_RUNTIME_VERSION}
COPY --from=onnxruntime-downloader ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${ONNX_RUNTIME}"

# Copy Libtorch
ARG LIBTORCH_VERSION
ENV LIBTORCH=/opt/libtorch/${LIBTORCH_VERSION}
COPY --from=torch-downloader ${LIBTORCH} ${LIBTORCH}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${LIBTORCH}"

# Copy TorchVision
ARG TORCHVISION_VERSION
ENV TORCHVISION=/opt/torchvision/${TORCHVISION_VERSION}
COPY --from=torchvision-downloader ${TORCHVISION} ${TORCHVISION}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${TORCHVISION}"

# Copy gRPC
ARG GRPC_VERSION
ENV GRPC=/opt/grpc/${GRPC_VERSION}
COPY --from=grpc-builder ${GRPC} ${GRPC}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GRPC}"

# Copy UCX and set other HPC-X runtime paths
ENV HPCX=/opt/hpcx
COPY --from=ucx-patcher ${HPCX}/ucx ${HPCX}/ucx
ENV PATH="${PATH}:${HPCX}/ucx/bin:${HPCX}/ucc/bin:${HPCX}/ompi/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${HPCX}/ucx"
# Constrain HPCX's ld config to Holoscan/Torch explicit dependencies,
# to prevent inadvertently picking up non-expected libraries
RUN echo "${HPCX}/ucx/lib" > /etc/ld.so.conf.d/hpcx.conf \
    && echo "${HPCX}/ucc/lib" >> /etc/ld.so.conf.d/hpcx.conf \
    && echo "${HPCX}/ompi/lib" >> /etc/ld.so.conf.d/hpcx.conf

# Copy GXF
ARG GXF_VERSION
ENV GXF=/opt/nvidia/gxf/${GXF_VERSION}
COPY --from=gxf-downloader ${GXF} ${GXF}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GXF}"
ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/gxf/${GXF_VERSION}/python"

# Setup Docker & NVIDIA Container Toolkit's apt repositories to enable DooD
# for packaging & running applications with the CLI
# Ref: Docker installation: https://docs.docker.com/engine/install/ubuntu/
# DooD (Docker-out-of-Docker): use the Docker (or Moby) CLI in your dev container to connect to
#  your host's Docker daemon by bind mounting the Docker Unix socket.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install NVIDIA Performance Libraries on arm64 dGPU platform
# as a runtime requirement for the Holoinfer `libtorch` backend (2.5.0).
ARG GPU_TYPE
RUN if [[ $(uname -m) = "aarch64" && ${GPU_TYPE} = "dgpu" ]]; then \
    curl -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb -O \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        nvpl-blas=0.2.0.1-* \
        nvpl-lapack=0.2.2.1-* \
    && apt-get purge -y cuda-keyring \
    && rm cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/* \
    ; fi

# APT INSTALLS
#  valgrind - dynamic analysis
#  clang-tidy - static analysis
#  xvfb - testing on headless systems
#  libx* - X packages
#  libvulkan-dev, glslang-tools - for Vulkan apps (Holoviz)
#  vulkan-validationlayers - for Vulkan validation layer (enabled for Holoviz in debug mode)
#  libwayland-dev, libxkbcommon-dev, pkg-config - GLFW compile dependency for Wayland support
#  libdecor-0-plugin-1-cairo - GLFW runtime dependency for Wayland window decorations
#  libegl1 - to run headless Vulkan apps
#  libopenblas0 - libtorch dependency
#  libv4l-dev - V4L2 operator dependency
#  v4l-utils - V4L2 operator utility
#  libpng-dev - torchvision dependency
#  libjpeg-dev - torchvision, v4l2 mjpeg dependency
#  docker-ce-cli - enable Docker DooD for CLI
#  docker-buildx-plugin - enable Docker DooD for CLI
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind="1:3.18.1-*" \
        clang-tidy="1:14.0-*" \
        xvfb="2:21.1.4-*" \
        libx11-dev="2:1.7.5-*" \
        libxcb-glx0="1.14-*" \
        libxcursor-dev="1:1.2.0-*" \
        libxi-dev="2:1.8-*" \
        libxinerama-dev="2:1.1.4-*" \
        libxrandr-dev="2:1.5.2-*" \
        libvulkan-dev="1.3.204.1-*" \
        glslang-tools="11.8.0+1.3.204.0-*" \
        vulkan-validationlayers="1.3.204.1-*" \
        libwayland-dev="1.20.0-*" \
        libxkbcommon-dev="1.4.0-*" \
        pkg-config="0.29.2-*" \
        libdecor-0-plugin-1-cairo="0.1.0-*" \
        libegl1="1.4.0-*" \
        libopenblas0="0.3.20+ds-*" \
        libv4l-dev="1.22.1-*" \
        v4l-utils="1.22.1-*" \
        libpng-dev="1.6.37-*" \
        libjpeg-turbo8-dev="2.1.2-*" \
        docker-ce-cli="5:25.0.3-*" \
        docker-buildx-plugin="0.12.1-*" \
    && rm -rf /var/lib/apt/lists/*

# PIP INSTALLS
#  mkl - dependency for libtorch plugin on x86_64 (match pytorch container version)
#  requirements.dev.txt
#    coverage - test coverage of python tests
#    pytest*  - testing
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
RUN if [ $(uname -m) = "x86_64" ]; then \
        python3 -m pip install --no-cache-dir \
            mkl==2021.1.1 \
        && \
        # Clean up duplicate libraries from mkl/tbb python wheel install which makes copies for symlinks.
        # Only keep the *.so.X libs, remove the *.so and *.so.X.Y libs
        # This can be removed once upgrading to an MKL pip wheel that fixes the symlinks
        find /usr/local/lib -maxdepth 1 -type f -regex '.*\/lib\(tbb\|mkl\).*\.so\(\.[0-9]+\.[0-9]+\)?' -exec rm -v {} +; \
    fi
COPY python/requirements.dev.txt /tmp/requirements.dev.txt
COPY python/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.dev.txt -r /tmp/requirements.txt

# Creates a home directory for docker-in-docker to store files temporarily in the container,
# necessary when running the holoscan CLI packager
ENV HOME=/home/holoscan
RUN mkdir -p $HOME && chmod 777 $HOME

############################################################################################
# Extra stage: igpu build image
# The iGPU CMake build depends on libnvcudla.so as well as libnvdla_compiler.so, which are
# part of the L4T BSP. As such, they should not be in the container, but mounted at runtime
# (which the nvidia container runtime handles). However, we need the symbols at build time
# for the TensorRT libraries to resolve. Since there is no stub library (unlike libcuda.so),
# we need to include them in our builder. We use a separate stage so that `run build` can
# use it if needed, but `run launch` (used to run apps in the container) doesn't need to.
############################################################################################
FROM build AS build-igpu
ARG GPU_TYPE
RUN if [ ${GPU_TYPE} = "igpu" ]; then \
        tmp_dir=$(mktemp -d) \
        && curl -S -# -L -o $tmp_dir/l4t_core.deb \
            https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-core/nvidia-l4t-core_36.1.0-20231206095146_arm64.deb \
        && curl -S -# -L -o $tmp_dir/l4t_cuda.deb \
            https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-cuda/nvidia-l4t-cuda_36.1.0-20231206095146_arm64.deb \
        && curl -S -# -L -o $tmp_dir/l4t_dla.deb \
            https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.1.0-20231206095146_arm64.deb \
        && dpkg -x $tmp_dir/l4t_core.deb / \
        && dpkg -x $tmp_dir/l4t_cuda.deb / \
        && dpkg -x $tmp_dir/l4t_dla.deb /; \
    fi
