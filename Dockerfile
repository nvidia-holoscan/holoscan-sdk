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
ARG ONNX_RUNTIME_VERSION=1.15.1_23.08
ARG LIBTORCH_VERSION=2.1.0_23.08
ARG TORCHVISION_VERSION=0.16.0_23.08
ARG VULKAN_SDK_VERSION=1.3.216.0
ARG GRPC_VERSION=1.54.2
ARG UCX_VERSION=1.15.0
ARG GXF_VERSION=4.0_20240409_bc03d9d
ARG MOFED_VERSION=23.10-2.1.3.1

############################################################
# Base image
############################################################
ARG GPU_TYPE=dgpu
FROM nvcr.io/nvidia/tensorrt:23.08-py3 AS dgpu_base
FROM nvcr.io/nvidia/tensorrt:23.12-py3-igpu AS igpu_base
FROM ${GPU_TYPE}_base AS base

ARG DEBIAN_FRONTEND=noninteractive

############################################################
# Variables
############################################################
ARG MAX_PROC=32

############################################################
# Build tools
############################################################
FROM base as build-tools

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
FROM base as ngc-cli-downloader

WORKDIR /opt/ngc-cli

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && curl -S -# -L -o ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -q ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && export ngc_exec=$(find . -type f -executable -name "ngc" | head -n1)

############################################################
# ONNX Runtime
############################################################
FROM base as onnxruntime-downloader
ARG ONNX_RUNTIME_VERSION

# Download ORT binaries from artifactory
# note: built with CUDA and TensorRT providers
WORKDIR /opt/onnxruntime
RUN curl -S -L -# -o ort.tgz \
    https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/onnxruntime/onnxruntime-${ONNX_RUNTIME_VERSION}-cuda-12.2-$(uname -m).tar.gz
RUN mkdir -p ${ONNX_RUNTIME_VERSION}
RUN tar -xf ort.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 2

############################################################
# Libtorch
############################################################
FROM base as torch-downloader
ARG LIBTORCH_VERSION
ARG GPU_TYPE

# Download libtorch binaries from artifactory
# note: extracted from nvcr.io/nvidia/pytorch:23.07-py3
WORKDIR /opt/libtorch/
RUN ARCH=$(uname -m) && if [ "$ARCH" = "aarch64" ]; then ARCH="${ARCH}-${GPU_TYPE}"; fi && \
    curl -S -# -o libtorch.tgz -L \
        https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/libtorch/libtorch-${LIBTORCH_VERSION}-${ARCH}.tar.gz
RUN mkdir -p ${LIBTORCH_VERSION}
RUN tar -xf libtorch.tgz -C ${LIBTORCH_VERSION} --strip-components 1
# Remove kineto from config to remove warning, not needed by holoscan
RUN find . -type f -name "*Config.cmake" -exec sed -i '/kineto/d' {} +

############################################################
# TorchVision
############################################################
FROM base as torchvision-downloader
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
# Vulkan SDK
#
# Use the SDK because we need the newer Vulkan headers and the newer shader compiler than provided
# by the Ubuntu deb packages. These are compile time dependencies, we still use the Vulkan loaded
# and the Vulkan validation layer as runtime components provided by Ubuntu packages because that's
# what the user will have on their installations.
############################################################
FROM build-tools as vulkansdk-builder
ARG VULKAN_SDK_VERSION

WORKDIR /opt/vulkansdk

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and e7ab9314build the aarch64 version from source.
RUN curl -S -# -O -L https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN if [ $(uname -m) = "aarch64" ]; then \
    cd ${VULKAN_SDK_VERSION} \
    && rm -rf x86_64 \
    && ./vulkansdk shaderc glslang headers; \
    fi

############################################################
# gRPC libraries and binaries
############################################################
FROM build-tools as grpc-builder
ARG GRPC_VERSION

WORKDIR /opt/grpc
RUN git clone --depth 1 --branch v${GRPC_VERSION} \
         --recurse-submodules --shallow-submodules \
         https://github.com/grpc/grpc.git src
RUN cmake -S src -B build -G Ninja \
         -D CMAKE_BUILD_TYPE=Release \
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
FROM mofed-installer as ucx-builder
ARG UCX_VERSION

# Clone
WORKDIR /opt/ucx/
RUN git clone --depth 1 --branch v${UCX_VERSION} https://github.com/openucx/ucx.git src

# Patch
WORKDIR /opt/ucx/src
RUN curl -L https://github.com/openucx/ucx/pull/9341.patch | git apply

# Prerequisites to build
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libtool="2.4.6-*" \
        automake="1:1.16.5-*" \
    && rm -rf /var/lib/apt/lists/*

# Build and install
RUN ./autogen.sh
WORKDIR /opt/ucx/build
RUN ../src/contrib/configure-release-mt --with-cuda=/usr/local/cuda-12 \
    --prefix=/opt/ucx/${UCX_VERSION}
RUN make -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` )) install

# Apply patches for import and run
WORKDIR /opt/ucx/${UCX_VERSION}
# patch cmake config
RUN sed -i "s|set(prefix.*)|set(prefix \"$(pwd)\")|" lib/cmake/ucx/ucx-targets.cmake
# patch rpath (relative to ORIGIN)
RUN patchelf --set-rpath '$ORIGIN' lib/libuc*.so*
RUN patchelf --set-rpath '$ORIGIN:$ORIGIN/..' lib/ucx/libuc*.so*
RUN patchelf --set-rpath '$ORIGIN/../lib' bin/*

############################################################
# GXF
############################################################
FROM base as gxf-builder
ARG GXF_VERSION

WORKDIR /opt/nvidia/gxf
RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=x86_64; fi \
    && curl -S -# -L -o gxf.tgz \
        https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/gxf/gxf_${GXF_VERSION}_holoscan-sdk_${ARCH}.tar.gz
RUN mkdir -p ${GXF_VERSION}
RUN tar xzf gxf.tgz -C ${GXF_VERSION} --strip-components 1

############################################################
# Build image (final)
############################################################
FROM mofed-installer as build

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

# Copy Vulkan SDK
ARG VULKAN_SDK_VERSION
ENV VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
COPY --from=vulkansdk-builder ${VULKAN_SDK}/x86_64/ ${VULKAN_SDK}
# We need to use the headers and shader compiler of the SDK but want to link against the
# Vulkan loader provided by the Ubuntu package. Therefore create a link in the SDK directory
# pointing to the system Vulkan loader library.
RUN rm -f ${VULKAN_SDK}/lib/libvulkan.so* \
    && ln -s /lib/$(uname -m)-linux-gnu/libvulkan.so.1 ${VULKAN_SDK}/lib/libvulkan.so
ENV PATH="${PATH}:${VULKAN_SDK}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${VULKAN_SDK}"

# Copy gRPC
ARG GRPC_VERSION
ENV GRPC=/opt/grpc/${GRPC_VERSION}
COPY --from=grpc-builder ${GRPC} ${GRPC}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GRPC}"

# Copy UCX
ARG UCX_VERSION
ENV UCX=/opt/ucx/${UCX_VERSION}
COPY --from=ucx-builder ${UCX} ${UCX}
ENV PATH="${PATH}:${UCX}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${UCX}"
# remove older version of UCX in hpcx install
RUN rm -rf /opt/hpcx/ucx /usr/local/ucx
RUN unset OPENUCX_VERSION
# required for gxf_ucx.so to find ucx
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${UCX}/lib"

# Copy GXF
ARG GXF_VERSION
ENV GXF=/opt/nvidia/gxf/${GXF_VERSION}
COPY --from=gxf-builder ${GXF} ${GXF}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GXF}"

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

# APT INSTALLS
#  valgrind - static analysis
#  xvfb - testing on headless systems
#  libx* - X packages
#  libvulkan1 - for Vulkan apps (Holoviz)
#  vulkan-validationlayers, spirv-tools - for Vulkan validation layer (enabled for Holoviz in debug mode)
#  libegl1 - to run headless Vulkan apps
#  libopenblas0 - libtorch dependency
#  libv4l-dev - V4L2 operator dependency
#  v4l-utils - V4L2 operator utility
#  libpng-dev - torchvision dependency
#  libjpeg-dev - torchvision dependency
#  docker-ce-cli - enable Docker DooD for CLI
#  docker-buildx-plugin - enable Docker DooD for CLI
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind="1:3.18.1-*" \
        xvfb="2:21.1.4-*" \
        libx11-dev="2:1.7.5-*" \
        libxcb-glx0="1.14-*" \
        libxcursor-dev="1:1.2.0-*" \
        libxi-dev="2:1.8-*" \
        libxinerama-dev="2:1.1.4-*" \
        libxrandr-dev="2:1.5.2-*" \
        libvulkan1="1.3.204.1-*" \
        vulkan-validationlayers="1.3.204.1-*" \
        spirv-tools="2022.1+1.3.204.0-*" \
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
FROM build as build-igpu
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
