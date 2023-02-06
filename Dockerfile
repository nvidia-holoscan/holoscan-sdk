# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG TRT_CONTAINER_TAG=22.03-py3
ARG ONNX_RUNTIME_VERSION=1.12.1
ARG VULKAN_SDK_VERSION=1.3.216.0

############################################################
# Base image
############################################################
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:${TRT_CONTAINER_TAG}
FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

# Install newer CMake (https://apt.kitware.com/)
RUN rm -r \
    /usr/local/bin/cmake \
    /usr/local/bin/cpack \
    /usr/local/bin/ctest \
    /usr/local/share/cmake-3.14
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt update \
    && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
    && apt install --no-install-recommends -y \
        kitware-archive-keyring \
        cmake-data=3.22.2-0kitware1ubuntu20.04.1 \
        cmake=3.22.2-0kitware1ubuntu20.04.1 \
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

RUN if [ $(uname -m) == "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && wget \
        -nv --show-progress --progress=bar:force:noscroll \
        -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -o ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && export ngc_exec=$(find . -type f -executable -name "ngc" | head -n1)

############################################################
# ONNYX Runtime
############################################################
FROM base as onnxruntime-downloader
ARG ONNX_RUNTIME_VERSION=1.12.1

WORKDIR /opt/onnxruntime

RUN if [ $(uname -m) == "aarch64" ]; then ARCH=aarch64; else ARCH=x64-gpu; fi \
    && ONNX_RUNTIME_NAME="onnxruntime-linux-${ARCH}-${ONNX_RUNTIME_VERSION}" \
    && wget \
        -nv --show-progress --progress=bar:force:noscroll \
        https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/${ONNX_RUNTIME_NAME}.tgz \
    && mkdir -p ${ONNX_RUNTIME_VERSION} \
    && tar -xzf ${ONNX_RUNTIME_NAME}.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 1

############################################################
# Vulkan SDK
############################################################
FROM base as vulkansdk-builder
ARG VULKAN_SDK_VERSION=1.3.216.0

WORKDIR /opt/vulkansdk

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and build the aarch64 version from source.
RUN wget -nv --show-progress --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN if [ $(uname -m) == "aarch64" ]; then \
    apt update \
    && apt install -y --no-install-recommends \
        libwayland-dev=1.18.0-1ubuntu0.1 \
        libx11-dev=2:1.6.9-2ubuntu1.2 \
        libxrandr-dev=2:1.5.2-0ubuntu1 \
    && cd ${VULKAN_SDK_VERSION} \
    && rm -rf x86_64 \
    && ./vulkansdk shaderc glslang headers loader layers; \
fi

############################################################
# dev image
############################################################
FROM base as dev

# Install apt & pip tools
RUN apt update \
    && apt install --no-install-recommends -y \
        patchelf \
        lcov=1.14-2 \
        python3-pytest \
        ninja-build=1.10.0-1build1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install \
        coverage==6.5.0

# Install apt & pip build dependencies
RUN apt update \
    && apt install --no-install-recommends -y \
        libgl-dev=1.3.2-1~ubuntu0.20.04.2 \
        libx11-dev=2:1.6.9-2ubuntu1.2 \
        libxcursor-dev=1:1.2.0-2 \
        libxext-dev=2:1.3.4-0ubuntu1 \
        libxi-dev=2:1.7.10-0ubuntu1 \
        libxinerama-dev=2:1.1.4-2 \
        libxrandr-dev=2:1.5.2-0ubuntu1 \
        libxxf86vm-dev=1:1.1.4-1build1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip run dependencies
# Note: cupy pypi wheels not available for arm64
RUN if [ $(uname -m) != "aarch64" ]; then \
    python3 -m pip install \
        cupy-cuda11x==11.3.0; \
    fi

## COPY BUILT/DOWNLOADED dependencies in previous stages
# Note: avoid LD_LIBRARY_PATH - https://www.hpc.dtu.dk/?page_id=1180

# Copy NGC CLI
ENV NGC_CLI=/opt/ngc-cli
COPY --from=ngc-cli-downloader ${NGC_CLI}/ngc-cli ${NGC_CLI}
ENV PATH="${PATH}:${NGC_CLI}"

# Copy ONNX Runtime
ARG ONNX_RUNTIME_VERSION
ENV ONNX_RUNTIME=/opt/onnxruntime/${ONNX_RUNTIME_VERSION}
COPY --from=onnxruntime-downloader ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${ONNX_RUNTIME}"

# Copy Vulkan SDK
ARG VULKAN_SDK_VERSION
ENV VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
COPY --from=vulkansdk-builder ${VULKAN_SDK}/x86_64/ ${VULKAN_SDK}
RUN mkdir -p /usr/share/vulkan/explicit_layer.d \
    && cp ${VULKAN_SDK}/etc/vulkan/explicit_layer.d/VkLayer_*.json /usr/share/vulkan/explicit_layer.d \
    && echo ${VULKAN_SDK}/lib > /etc/ld.so.conf.d/vulkan.conf
ENV PATH="${PATH}:${VULKAN_SDK}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${VULKAN_SDK}"
