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
FROM nvcr.io/nvidia/tensorrt:${TRT_CONTAINER_TAG}
ENV DEBIAN_FRONTEND=noninteractive

# PREREQUISITES

## NGC CLI
ARG NGC_CLI_ORG=nvidia/clara-holoscan
WORKDIR /etc/ngc
RUN if [ $(uname -m) == "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && wget \
        -nv --show-progress --progress=bar:force:noscroll \
        -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
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

# DIRECT DEPENDENCIES

# - This variable is consumed by all depencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time
ARG CMAKE_BUILD_TYPE=Release

## AJA NTV2 SDK
ARG AJA_NTV2_TAG=cmake-exports
WORKDIR /tmp/ajantv2
RUN git clone https://github.com/ibstewart/ntv2.git src -b ${AJA_NTV2_TAG} \
    && cmake -S src -B build -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
        -D AJA_BUILD_APPS:BOOL=OFF \
        -D AJA_BUILD_DOCS:BOOL=OFF \
        -D AJA_BUILD_DRIVER:BOOL=OFF \
        -D AJA_BUILD_LIBS:BOOL=ON \
        -D AJA_BUILD_PLUGINS:BOOL=OFF \
        -D AJA_BUILD_QA:BOOL=OFF \
        -D AJA_BUILD_TESTS:BOOL=OFF \
        -D AJA_INSTALL_HEADERS:BOOL=ON \
        -D AJA_INSTALL_SOURCES:BOOL=OFF \
        -D CMAKE_INSTALL_PREFIX:PATH=$PWD/unecessary-copy-to-remove \
    && cmake --build build -j \
    && cmake --install build --prefix /opt/ajantv2 \
    && cd .. && rm -rf ajantv2

## glad
ARG GLAD_TAG=v0.1.36
WORKDIR /tmp/glad
RUN git clone https://github.com/Dav1dde/glad.git src -b ${GLAD_TAG}  \
    && cmake -S src -B build -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
        -D GLAD_INSTALL:BOOL=ON \
    && cmake --build build -j \
    && cmake --install build --prefix /opt/glad \
    && cd .. && rm -rf glad

## glfw
ARG GLFW_TAG=3.2.1
WORKDIR /tmp/glfw
RUN git clone https://github.com/glfw/glfw.git src -b ${GLFW_TAG} \
    && cmake -S src -B build -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
        -D GLFW_BUILD_DOCS:BOOL=OFF \
        -D GLFW_BUILD_EXAMPLES:BOOL=OFF \
        -D GLFW_BUILD_TESTS:BOOL=OFF \
    && cmake --build build -j \
    && cmake --install build --prefix /opt/glfw \
    && cd .. && rm -rf glfw

## GXF
ARG GXF_TAG=2.4.2-f90116f2
WORKDIR /tmp/gxf
RUN --mount=type=secret,id=NGC_CLI_API_KEY \
    export NGC_CLI_API_KEY="$(cat /run/secrets/NGC_CLI_API_KEY)" \
    && if [ $(uname -m) == "aarch64" ]; then ARCH=arm64; else ARCH="x86_64"; fi \
    && export ngc_exec=$(find /etc/ngc -type f -executable -name "ngc" | head -n1) \
    && $ngc_exec registry resource download-version "${NGC_CLI_ORG}/gxf_${ARCH}_holoscan_sdk:${GXF_TAG}" \
    && mkdir -p /opt/gxf \
    && tar -zxf gxf*/*.tar.gz -C /opt/gxf/ --strip-components=1 \
    && cd .. && rm -rf gxf

## nanovg
ARG NANOVG_TAG=5f65b43
WORKDIR /tmp/nanovg
ADD cmake/patches/nanovg/* patches/
RUN git clone https://github.com/memononen/nanovg.git src \
    && mv patches/* src/ \
    && pushd src \
    && git checkout ${NANOVG_TAG} \
    && popd \
    && cmake -S src -B build -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
    && cmake --build build -j \
    && cmake --install build --prefix /opt/nanovg \
    && cd .. && rm -rf nanovg

## yaml-cpp
ARG YAML_CPP_TAG=yaml-cpp-0.6.3
WORKDIR /tmp/yaml-cpp
RUN git clone https://github.com/jbeder/yaml-cpp.git src -b ${YAML_CPP_TAG} \
    && cmake -S src -B build -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
        -D YAML_CPP_BUILD_CONTRIB:BOOL=OFF \
        -D YAML_CPP_BUILD_TESTS:BOOL=OFF \
        -D YAML_CPP_BUILD_TOOLS:BOOL=OFF \
    && cmake --build build -j \
    && cmake --install build --prefix /opt/yaml-cpp \
    && cd .. && rm -rf yaml-cpp

# Default entrypoint
WORKDIR /workspace/holoscan-sdk
