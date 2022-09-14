# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TRT_CONTAINER_TAG=22.03-py3
ARG VULKAN_SDK_VERSION=1.3.216.0

FROM nvcr.io/nvidia/tensorrt:${TRT_CONTAINER_TAG} as base
ENV DEBIAN_FRONTEND=noninteractive

# PREREQUISITES

## NGC CLI
ARG NGC_CLI_ORG=nvidia/clara-holoscan
WORKDIR /etc/ngc
RUN if [ $(uname -m) == "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && for i in {1..5}; do \
            wget \
                -nv --show-progress --progress=bar:force:noscroll \
                -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
            && break \
            || sleep 5; \
        done \
    && unzip -o ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && export ngc_exec=$(find . -type f -executable -name "ngc" | head -n1) \
    && chmod u+x $ngc_exec

## Tooling
RUN apt update \
    && apt install --no-install-recommends -y \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

## CMake
RUN rm -r \
    /usr/local/bin/cmake \
    /usr/local/bin/cpack \
    /usr/local/bin/ctest \
    /usr/local/share/cmake-3.14
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ focal main" \
    && apt update \
    && apt install --no-install-recommends -y \
        cmake-data=3.22.2-0kitware1ubuntu20.04.1 \
        cmake=3.22.2-0kitware1ubuntu20.04.1 \
    && rm -rf /var/lib/apt/lists/*

## X11 & GL
RUN apt update \
    && apt install --no-install-recommends -y \
        libgl1=1.3.2-1~ubuntu0.20.04.2 \
        libx11-dev=2:1.6.9-2ubuntu1.2 \
        libxcursor-dev=1:1.2.0-2 \
        libxrandr-dev=2:1.5.2-0ubuntu1 \
        libxinerama-dev=2:1.1.4-2 \
        libxi-dev=2:1.7.10-0ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

## Holoviz build dependencies
RUN apt update \
    && apt install --no-install-recommends -y \
        libxxf86vm-dev=1:1.1.4-1build1 \
        libxext-dev=2:1.3.4-0ubuntu1 \
        libgl-dev=1.3.2-1~ubuntu0.20.04.2 \
    && rm -rf /var/lib/apt/lists/*


# DIRECT DEPENDENCIES

# - This variable is consumed by all depencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time
ARG CMAKE_BUILD_TYPE=Release

## Vulkan SDK
# Use a multi-stage build to reduce the docker image size
FROM base as vulkansdk

ARG VULKAN_SDK_VERSION
ARG MAX_WORKERS

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and build the aarch64 version from source.
WORKDIR /tmp/vulkansdk
RUN wget --inet4-only -nv --show-progress --progress=dot:giga https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    && tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    && rm vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    && if [ $(uname -m) == "aarch64" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            cmake-data=3.22.2-0kitware1ubuntu20.04.1 \
            cmake=3.22.2-0kitware1ubuntu20.04.1 \
            libglm-dev libxcb-dri3-0 libxcb-present0 libpciaccess0 \
            libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev=2:1.6.9-2ubuntu1.2 g++ gcc \
            libmirclient-dev libwayland-dev libxrandr-dev=2:1.5.2-0ubuntu1 libxcb-randr0-dev libxcb-ewmh-dev \
            git python python3 bison libx11-xcb-dev liblz4-dev libzstd-dev python3-distutils \
            qt5-default ocaml-core ninja-build pkg-config libxml2-dev wayland-protocols \
        && cd ${VULKAN_SDK_VERSION} \
        && rm -rf x86_64 \
        && echo "Building aarch64 version from source consumes much memory so limit # of workers by setting environment variable 'MAKEFLAGS' to '-j<MAX_WORKERS>'." \
        && if [ -n "${MAX_WORKERS}" ]; then NUM_WORKERS=${MAX_WORKERS}; else NUM_WORKERS=$(nproc); fi \
        && export MAKEFLAGS="-j${NUM_WORKERS}" \
        && echo "MAKEFLAGS=$MAKEFLAGS" \
        && ./vulkansdk shaderc glslang headers loader; \
    fi

FROM base as dependencies

ARG VULKAN_SDK_VERSION

COPY --from=vulkansdk /tmp/vulkansdk/${VULKAN_SDK_VERSION}/x86_64/ /opt/vulkansdk/${VULKAN_SDK_VERSION}
ENV VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
ENV PATH="$VULKAN_SDK/bin:${PATH}"

ENV PATH="${PATH}:/etc/ngc/ngc-cli"

# Default entrypoint
WORKDIR /workspace/holoscan-sdk
