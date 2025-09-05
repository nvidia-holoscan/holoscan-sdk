# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG ONNX_RUNTIME_VERSION=1.22.1
ARG ONNX_RUNTIME_STRATEGY=downloader # or builder
ARG LIBTORCH_VERSION=2.8.0
ARG GRPC_VERSION=1.54.2
ARG GXF_CU12_VERSION=5.1.0_20250820_2ac8c610f_holoscan-sdk-cu12
ARG GXF_CU13_VERSION=5.1.0_20250820_2ac8c610f_holoscan-sdk-cu13
ARG DOCA_VERSION=3.0.0
ARG TENSORRT_CU12_VERSION=10.3  # TRT 10.3 is the last version that supports CUDA 12 on sbsa 22.04
ARG TENSORRT_CU13_VERSION=10.13

############################################################
# Base image
# Notes:
# - iGPU base is only used for Orin, not Thor
# - no 22.04 support for TRT+cu13 on sbsa, hence 24.04 base
# - select preset base images with GPU_TYPE and CUDA_MAJOR
# - setting BASE_IMAGE will ignore GPU_TYPE and CUDA_MAJOR
############################################################
ARG GPU_TYPE=dgpu
ARG CUDA_MAJOR=12
ARG BASE_IMAGE=${TARGETARCH}-${GPU_TYPE}_cu${CUDA_MAJOR}_base
FROM nvcr.io/nvidia/cuda:12.6.3-base-ubuntu22.04 AS amd64-dgpu_cu12_base
FROM nvcr.io/nvidia/cuda:13.0.0-base-ubuntu22.04 AS amd64-dgpu_cu13_base
FROM nvcr.io/nvidia/cuda:12.6.3-base-ubuntu22.04 AS arm64-dgpu_cu12_base
FROM nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 AS arm64-dgpu_cu13_base
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime AS arm64-igpu_cu12_base
FROM ${BASE_IMAGE} AS base

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]

# Conditionally configure apt caching behavior
ARG ENABLE_APT_CACHING=true
RUN if [ "${ENABLE_APT_CACHING}" = "true" ]; then \
        echo "APT Caching enabled: Disabling docker-clean and enabling Keep-Downloaded-Packages."; \
        DOCKER_CLEAN_CONF="/etc/apt/apt.conf.d/docker-clean"; \
        if [ -f "${DOCKER_CLEAN_CONF}" ]; then \
            mv "${DOCKER_CLEAN_CONF}" "${DOCKER_CLEAN_CONF}.disabled"; \
        fi; \
        echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/99-keep-archives; \
    else \
        echo "APT Caching disabled: Adding extra Post-Invoke cleanup hook."; \
        echo 'DPkg::Post-Invoke "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin /var/lib/apt/lists/* || true";' > /etc/apt/apt.conf.d/99-custom-cleanup; \
    fi

# Common variables
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG TARGETARCH
ARG GPU_TYPE
ARG CUDA_MAJOR

# Install basic tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update && apt-get install --no-install-recommends -y \
        curl \
        unzip \
        patch

# Install L4T APT repo on iGPU base
RUN if [ "${GPU_TYPE}" = "igpu" ]; then \
        L4T_APT_REPO_URL="https://repo.download.nvidia.com/jetson"; \
        L4T_APT_REPO_KEY="$L4T_APT_REPO_URL/jetson-ota-public.asc"; \
        L4T_APT_REPO_FILE="/usr/share/keyrings/jetson-ota-archive-keyring.gpg"; \
        L4T_APT_SOURCE_FILE="/etc/apt/sources.list.d/nvidia-l4t-apt-source.list"; \
        L4T_APT_BRANCH="r36.4"; \
        curl -sSL "$L4T_APT_REPO_KEY" | gpg --dearmor -o "$L4T_APT_REPO_FILE"; \
        echo "deb [signed-by=$L4T_APT_REPO_FILE] $L4T_APT_REPO_URL/common/ $L4T_APT_BRANCH main" > "$L4T_APT_SOURCE_FILE"; \
    fi

############################################################
# Python base
############################################################
FROM base AS python-base

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Ensure we use Python 3.12
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-dev \
    && apt purge -y \
        python3-pip \
        software-properties-common \
    && apt-get autoremove --purge -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

############################################################
# Build tools
############################################################
FROM python-base AS build-tools

# Install build tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
    && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
        | gpg --dearmor -o "$KW_KEYRING" \
    && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && rm "$KW_KEYRING" \
    && apt-get install --no-install-recommends -y \
        kitware-archive-keyring \
        cmake="3.*" \
        cmake-data="3.*" \
        build-essential \
        patchelf \
        ninja-build \
        git

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# Help limit the number of cores used for downstream builds
ARG MAX_PROC=16

############################################################
# CUDA dev
############################################################
FROM build-tools AS cuda-dev

# To inform users about deleted libraries
COPY  --chmod=755 <<'EOF' /usr/local/bin/cleanup_unwanted_libs
#!/bin/bash
DELETED_LIBS_LOG="/var/log/holoscan-deleted-libs.log"
DELETED_LIB_MSG="This file was deleted to optimize the size of the holoscan container (saved %s).
If you need this library, re-install it with: 'apt-get install --reinstall %s'.
The full list of deleted libraries is recorded in '${DELETED_LIBS_LOG}'."
for pattern in "$@"; do
    for file in $pattern; do
        if [ -f "${file}" ] && [ ! -L "${file}" ]; then
            lib_size=$(stat -c %s "${file}")
            lib_size_human=$(numfmt --to=iec --suffix=B ${lib_size})
            pkg=$(dpkg -S "${file}" 2>/dev/null | cut -d: -f1 || echo 'unknown')
            echo "${pkg} ${file} ${lib_size} ${lib_size_human}" | tee -a "${DELETED_LIBS_LOG}"
            printf "${DELETED_LIB_MSG}" "${lib_size_human}" "${pkg}" > "${file}"
        fi
    done
done
EOF

#  nvcc: needed by holoviz, holoinfer (cuda kernels)
#  cudart-dev: needed by holoscan core
#  nvrtc-dev: needed by holoscan core, and cupy (runtime only)
#  libnpp-dev: needed by bayer_demosaic, format_converter
#  libcublas-dev: needed by matx
#  libcufft-dev: needed by matx
#  libcurand-dev: needed by matx
#  libcusolver-dev: needed by matx
#  libcusparse-dev: needed by matx
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter="-") \
    && apt-get install --no-install-recommends -y \
        cuda-nvcc-${CUDA_MAJOR_MINOR} \
        cuda-cudart-dev-${CUDA_MAJOR_MINOR} \
        cuda-nvrtc-dev-${CUDA_MAJOR_MINOR} \
        libnpp-dev-${CUDA_MAJOR_MINOR} \
        libcublas-dev-${CUDA_MAJOR_MINOR} \
        libcufft-dev-${CUDA_MAJOR_MINOR} \
        libcurand-dev-${CUDA_MAJOR_MINOR} \
        libcusolver-dev-${CUDA_MAJOR_MINOR} \
        libcusparse-dev-${CUDA_MAJOR_MINOR} \
    && echo "-- Deleting unused static libs:" \
    && packages=$(dpkg -l | grep -e cuda-nvrtc -e libnpp | awk '{print $2}') \
    && static_libs=$(dpkg -L $packages | grep '\.a$' || true) \
    && cleanup_unwanted_libs $static_libs

############################################################
# Inference dev
############################################################
FROM cuda-dev AS infer-dev
ARG TENSORRT_CU12_VERSION
ARG TENSORRT_CU13_VERSION

#  libnvinfer*0dev: needed by holoinfer (trt)
#  libnvonnxparsers: needed by holoinfer (trt)
#  libcudnn: needed by libtorch, onnxruntime
#  cuda-nvtx: needed by libtorch/caffe2
#  libcublas: needed by libtorch, onnxruntime, cupy-cuda
#  libcufft: needed by libtorch, onnxruntime, cupy
#  libcurand: needed by libtorch, cupy-cuda
#  libcusparse: needed by libtorch, cupy-cuda
#  libcusparselt0: needed by libtorch (x86_64 and aarch64 dgpu/sbsa only)
#  libcusolver: needed by libtorch, cupy-cuda
#  libcufile: needed by libtorch
#  libnvjitlink: needed by libtorch, cupy-cuda
#  cuda-cupti: needed by libtorch
#  libnccl: needed by libtorch (x86_64 and aarch64 dgpu/sbsa only), cupy-cuda (optional)
#  nvpl-*: needed by libtorch (aarch64 dgpu/sbsa only)
#  libopenblas0 - needed by libtorch (aarch64 igpu/jetson only)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter="-") \
    && if [ ${GPU_TYPE} = "dgpu" ]; then \
        CONDITIONAL_LIBS="libnccl2 libcusparselt0"; \
        if [ $(uname -m) = "aarch64" ]; then \
            CONDITIONAL_LIBS="${CONDITIONAL_LIBS} nvpl-blas nvpl-lapack"; \
        fi; \
    else \
        CONDITIONAL_LIBS="libopenblas0"; \
    fi \
    && TRT_VERSION_VAR_NAME="TENSORRT_CU${CUDA_MAJOR}_VERSION" \
    && TRT_VERSION=$(apt-cache madison libnvinfer10 | grep "${!TRT_VERSION_VAR_NAME}" | grep "+cuda${CUDA_MAJOR}" | head -n 1 | awk '{print $3}') \
    && apt-get install --no-install-recommends -y \
        libnvonnxparsers-dev="${TRT_VERSION}" \
        libnvonnxparsers10="${TRT_VERSION}" \
        libnvinfer-plugin-dev="${TRT_VERSION}" \
        libnvinfer-headers-plugin-dev="${TRT_VERSION}" \
        libnvinfer-plugin10="${TRT_VERSION}" \
        libnvinfer-dev="${TRT_VERSION}" \
        libnvinfer-headers-dev="${TRT_VERSION}" \
        libnvinfer10="${TRT_VERSION}" \
        libcudnn9-cuda-${CUDA_MAJOR} \
        cuda-nvtx-${CUDA_MAJOR_MINOR} \
        libcublas-${CUDA_MAJOR_MINOR} \
        libcufft-${CUDA_MAJOR_MINOR} \
        libcurand-${CUDA_MAJOR_MINOR} \
        libcusparse-${CUDA_MAJOR_MINOR} \
        libcusolver-${CUDA_MAJOR_MINOR} \
        libcufile-${CUDA_MAJOR_MINOR} \
        libnvjitlink-${CUDA_MAJOR_MINOR} \
        cuda-cupti-${CUDA_MAJOR_MINOR} \
        ${CONDITIONAL_LIBS} \
    && echo "-- Deleting unused static libs:" \
    && packages=$(dpkg -l | grep -e libnvinfer -e libnvonnxparsers -e libcudnn -e cuda-nvtx -e cuda-cupti | awk '{print $2}') \
    && static_libs=$(dpkg -L $packages | grep '\.a$' || true) \
    && cleanup_unwanted_libs $static_libs \
    && echo "-- Deleting large unused libs:" \
    && SYSTEM_LIBS_ROOT="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)" \
    && cleanup_unwanted_libs \
        "${SYSTEM_LIBS_ROOT}/libnvinfer_builder_resource_win.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_cnn.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_engines_precompiled.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_engines_runtime_compiled.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_adv.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_ops.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_heuristic.*"

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
# ONNX Runtime (Source)
############################################################
FROM build-tools AS onnxruntime-src
ARG ORT_DIR=/opt/onnxruntime
ARG ONNX_RUNTIME_VERSION

# Clone
RUN git clone \
  --single-branch \
  --branch "v${ONNX_RUNTIME_VERSION}" \
  --recursive \
  https://github.com/Microsoft/onnxruntime \
  ${ORT_DIR}/src

# Apply patch for CUDA 12.9, TensorRT 10.11+, CUTLASS 3.9.2, and CUDA 13.0 support (ORT < 1.23)
WORKDIR ${ORT_DIR}/src
RUN ORT_REPO_PREFIX=https://github.com/microsoft/onnxruntime/commit/ && \
    curl -sSL ${ORT_REPO_PREFIX}ed7c234b2535.patch | git apply -v && \
    curl -sSL ${ORT_REPO_PREFIX}8983424d9a8d.patch | git apply -v && \
    curl -sSL ${ORT_REPO_PREFIX}9dad9af9f9b4.patch | git apply -v && \
    curl -sSL ${ORT_REPO_PREFIX}a2bd54bc8c59.patch | git apply -v && \
    curl -sSL ${ORT_REPO_PREFIX}7a6cef6fe367.patch | git apply -v --include=onnxruntime/contrib_ops/cuda/moe/ft_moe/moe_kernel.cu && \
    curl -sSL ${ORT_REPO_PREFIX}7c18d896b033.patch | git apply -v

############################################################
# ONNX Runtime (Build)
############################################################
FROM infer-dev AS onnxruntime-builder
ARG ORT_DIR=/opt/onnxruntime
ARG ONNX_RUNTIME_VERSION

# Build dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter='-') \
    && apt-get install --no-install-recommends -y \
        libcublas-dev-${CUDA_MAJOR_MINOR} \
        libcufft-dev-${CUDA_MAJOR_MINOR} \
        libcurand-dev-${CUDA_MAJOR_MINOR} \
        libcusparse-dev-${CUDA_MAJOR_MINOR} \
        libcudnn9-dev-cuda-${CUDA_MAJOR}

# Create user for non-root build
ARG BUILD_UID=1000
RUN id -u $BUILD_UID &>/dev/null || adduser --gecos 'ORT Build User' --disabled-password tmp_user --uid $BUILD_UID
RUN mkdir -p $ORT_DIR/build && chown -R $BUILD_UID $ORT_DIR/build
RUN mkdir -p $ORT_DIR/$ONNX_RUNTIME_VERSION && chown -R $BUILD_UID $ORT_DIR/$ONNX_RUNTIME_VERSION
USER $BUILD_UID

# Build
#
# Note: Using ORT with the TensorRT execution provider is highly advantageous:
# 1. Significantly lower latency and higher throughput compared to ORT+CUDA only.
# 2. If an ONNX layer cannot be converted to TensorRT, it falls back to CUDA,
#    while other layers can still be optimized with TRT.
# Given that TensorRT is already a requirement for the Holoscan SDK/HoloInfer,
# it is recommended to use it as an execution provider when running inference with ORT.
#
# Build references:
#   https://onnxruntime.ai/docs/build/eps.html#tensorrt
#   https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.tensorrt
#   https://github.com/triton-inference-server/onnxruntime_backend/blob/main/tools/gen_ort_dockerfile.py#L77
COPY scripts/get_cmake_cuda_archs.py ${ORT_DIR}/scripts/get_cmake_cuda_archs.py
RUN --mount=type=bind,from=onnxruntime-src,source=${ORT_DIR}/src,target=${ORT_DIR}/src \
    ORT_BUILD_THREADS=6 \
    && ORT_CUDA_ARCHS=$(${ORT_DIR}/scripts/get_cmake_cuda_archs.py all --min 75 --verbose) \
    && echo "ORT_CUDA_ARCHS: ${ORT_CUDA_ARCHS}" \
    && git config --global --add safe.directory ${ORT_DIR}/src \
    && ${ORT_DIR}/src/build.sh \
    --update \
    --skip_submodule_sync \
    --build \
    --build_shared_lib \
    --build_dir ${ORT_DIR}/build \
    --skip_tests \
    --parallel ${ORT_BUILD_THREADS} \
    --nvcc_threads 1 \
    --use_tensorrt \
    --use_tensorrt_builtin_parser \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/$(dpkg-architecture -qDEB_TARGET_MULTIARCH)/ \
    --tensorrt_home /usr/lib/$(dpkg-architecture -qDEB_TARGET_MULTIARCH)/ \
    --config Release \
    --cmake_extra_defines \
        "CMAKE_CUDA_ARCHITECTURES=${ORT_CUDA_ARCHS}" \
        "CMAKE_CXX_FLAGS=-Wno-deprecated-declarations" \
        "CMAKE_CUDA_FLAGS=-Xcompiler -Wno-deprecated-declarations" # error: ‘longlong4’ is deprecated: use longlong4_16a or longlong4_32a

# Install
RUN --mount=type=bind,from=onnxruntime-src,source=${ORT_DIR}/src,target=${ORT_DIR}/src \
    cmake --install ${ORT_DIR}/build/Release --prefix ${ORT_DIR}/${ONNX_RUNTIME_VERSION}
RUN --mount=type=bind,from=onnxruntime-src,source=${ORT_DIR}/src,target=${ORT_DIR}/src \
    cp \
    ${ORT_DIR}/src/LICENSE \
    ${ORT_DIR}/src/docs/Privacy.md \
    ${ORT_DIR}/src/README.md \
    ${ORT_DIR}/src/ThirdPartyNotices.txt \
    ${ORT_DIR}/src/VERSION_NUMBER \
    ${ORT_DIR}/${ONNX_RUNTIME_VERSION}

############################################################
# ONNX Runtime (Download)
############################################################
FROM base AS onnxruntime-downloader
ARG ONNX_RUNTIME_VERSION

# Download ORT binaries from artifactory
WORKDIR /opt/onnxruntime
RUN CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2) \
    && curl -S -L -# -o ort.tgz \
        https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/onnxruntime/onnxruntime-${ONNX_RUNTIME_VERSION}-cuda-${CUDA_MAJOR_MINOR}-$(uname -m).tar.gz
RUN mkdir -p ${ONNX_RUNTIME_VERSION}
RUN tar -xf ort.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 2 --no-same-owner --no-same-permissions

############################################################
# ONNX Runtime (Build or Download)
############################################################
FROM onnxruntime-${ONNX_RUNTIME_STRATEGY} AS onnxruntime

############################################################
# Libtorch
############################################################
FROM python-base AS libtorch-downloader
ARG LIBTORCH_VERSION

# Install torch wheel
ARG TORCH_WHL_DIR=/tmp/torch-whl
RUN --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    if [ "$GPU_TYPE" = "dgpu" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/test"; \
        TORCH_VERSION_ESCAPED=$(echo ${LIBTORCH_VERSION} | sed 's|\.|\\.|g'); \
        TORCH_WHL_VERSION=$(python3 -m pip index versions --index-url ${INDEX_URL} torch | \
            grep -oE "${TORCH_VERSION_ESCAPED}\+cu${CUDA_MAJOR}[0-9]" | \
            head -n 1); \
        if [ -z "${TORCH_WHL_VERSION}" ]; then \
            echo "Error: Could not find a matching torch version for torch ${LIBTORCH_VERSION} and CUDA ${CUDA_VERSION}." >&2; \
            exit 1; \
        fi; \
        echo "Found torch version: ${TORCH_WHL_VERSION}, installing..."; \
        python3 -m pip install \
            --no-deps \
            --target ${TORCH_WHL_DIR} \
            --index-url ${INDEX_URL} \
            torch==${TORCH_WHL_VERSION}; \
    else \
        python3 -m pip install \
            --no-deps \
            --python-version 3.10 \
            --target ${TORCH_WHL_DIR} \
            --index-url "https://pypi.jetson-ai-lab.io/jp6/cu126" \
            torch=="${LIBTORCH_VERSION}"; \
    fi

# Copy libtorch C++
WORKDIR /opt/libtorch/${LIBTORCH_VERSION}
RUN TORCH_INSTALL="${TORCH_WHL_DIR}/torch" \
    && echo "-- Copying libtorch ${LIBTORCH_VERSION} from ${TORCH_INSTALL} to $(pwd)" \
    && mkdir -p lib \
    && cp -v \
        "${TORCH_INSTALL}/lib/libc10.so" \
        "${TORCH_INSTALL}/lib/libc10_cuda.so" \
        "${TORCH_INSTALL}/lib/libshm.so" \
        "${TORCH_INSTALL}/lib/libtorch"* \
        lib/ \
    && if [ "$GPU_TYPE" = "dgpu" ]; then \
        cp -v "${TORCH_INSTALL}/lib/libgomp"* lib/ \
        && if [ $(uname -m) = "aarch64" ]; then \
            cp -v "${TORCH_INSTALL}/lib/libarm_"* lib/; \
        fi; \
    fi \
    && cp -rv "${TORCH_INSTALL}/bin" bin \
    && cp -rv "${TORCH_INSTALL}/share" share \
    && cp -rv "${TORCH_INSTALL}/include" include

# Patch step to remove kineto from config to remove warning, not needed by holoscan
RUN find . -type f -name "*Config.cmake" -exec sed -i '/kineto/d' {} +

############################################################################################
# DLA (iGPU drivers)
############################################################################################
FROM base AS dla-downloader

WORKDIR /opt/nvidia/dla
RUN curl -S -# -L -o l4t_core.deb \
    http://l4t-repo.nvidia.com/t234/pool/main/n/nvidia-l4t-core/nvidia-l4t-core_36.4.6-20250515220842_arm64.deb
RUN curl -S -# -L -o l4t_cuda.deb \
    http://l4t-repo.nvidia.com/t234/pool/main/n/nvidia-l4t-cuda/nvidia-l4t-cuda_36.4.6-20250515220842_arm64.deb
RUN curl -S -# -L -o l4t_dla.deb \
    http://l4t-repo.nvidia.com/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.4.6-20250515220842_arm64.deb

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
         -D gRPC_BUILD_TESTS=OFF \
         -D gRPC_BUILD_CSHARP_EXT=OFF \
         -D gRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
         -D gRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
         -D gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
         -D gRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
         -D gRPC_BUILD_GRPC_RUBY_PLUGIN=OFF
RUN cmake --build build -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN cmake --install build --prefix /opt/grpc/${GRPC_VERSION}

############################################################
# GXF
############################################################
FROM base AS gxf-downloader
ARG GXF_CU12_VERSION
ARG GXF_CU13_VERSION
ARG CUDA_MAJOR

WORKDIR /opt/nvidia/gxf
RUN if [ "${CUDA_MAJOR}" = "13" ]; then \
        GXF_VERSION="${GXF_CU13_VERSION}"; \
    else \
        GXF_VERSION="${GXF_CU12_VERSION}"; \
    fi; \
    curl -S -# -L -o gxf.tgz \
        "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/gxf/gxf_${GXF_VERSION}_$(uname -m).tar.gz"; \
    if [ $(stat -c %s gxf.tgz) -lt 1000 ]; then \
        echo "ERROR: Downloaded gxf.tgz is too small. Download may have failed."; \
        exit 1; \
    fi
RUN tar xzf gxf.tgz --strip-components 1 --no-same-owner --no-same-permissions

############################################################
# APT repository configs
############################################################
FROM base AS apt-repo-config

# Setup DOCA APT repository
ARG DOCA_VERSION
RUN DOCA_ARCH=$(uname -m); \
    if [ "${DOCA_ARCH}" = "aarch64" ]; then \
        DOCA_ARCH="arm64-sbsa"; \
    fi \
    && DOCA_REPO_ROOT="https://linux.mellanox.com/public/repo/doca" \
    && DOCA_HOSTNAME=$(echo "${DOCA_REPO_ROOT}" | sed 's|https://\([^/]*\).*|\1|') \
    && DISTRO=$(. /etc/os-release && echo "$ID$VERSION_ID") \
    && DOCA_URL="${DOCA_REPO_ROOT}/${DOCA_VERSION}/${DISTRO}/${DOCA_ARCH}/" \
    && DOCA_GPG_KEY="GPG-KEY-Mellanox.pub" \
    && DOCA_GPG_KEY_PATH="/etc/apt/trusted.gpg.d/${DOCA_GPG_KEY}" \
    && curl -fsSL ${DOCA_URL}/${DOCA_GPG_KEY} | gpg --dearmor -o ${DOCA_GPG_KEY_PATH} \
    && echo "deb [signed-by=${DOCA_GPG_KEY_PATH}] ${DOCA_URL} ./" \
        > /etc/apt/sources.list.d/doca.list \
    && DOCA_HOSTNAME=$(echo "${DOCA_REPO_ROOT}" | sed 's|https://\([^/]*\).*|\1|') \
    && echo "Package: *\nPin: origin \"${DOCA_HOSTNAME}\"\nPin-Priority: 800" \
        > /etc/apt/preferences.d/doca-pin

############################################################
# Build image
############################################################
FROM infer-dev AS build-generic

# Setup apt repositories
COPY --from=apt-repo-config /etc/apt/keyrings/ /etc/apt/keyrings/
COPY --from=apt-repo-config /etc/apt/sources.list.d/ /etc/apt/sources.list.d/
COPY --from=apt-repo-config /etc/apt/preferences.d/ /etc/apt/preferences.d/
COPY --from=apt-repo-config /etc/apt/trusted.gpg.d/ /etc/apt/trusted.gpg.d/

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
#  libv4l-dev - V4L2 operator dependency
#  v4l-utils - V4L2 operator utility
#  libjpeg-turbo8-dev - (8.0) v4l2 mjpeg dependency
#  ucx-*: needed for distributed apps (holoscan core) - comes from the DOCA repository
#  ibverbs* rdma*: needed for ConnectX RDMA support for ucx
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind \
        clang-tidy \
        xvfb \
        libx11-dev \
        libxcb-glx0 \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        libvulkan-dev \
        glslang-tools \
        vulkan-validationlayers \
        libwayland-dev \
        libxkbcommon-dev \
        pkg-config \
        libdecor-0-plugin-1-cairo \
        libegl1 \
        libv4l-dev \
        v4l-utils \
        libjpeg-turbo8-dev \
        ucx ucx-cuda \
        ibverbs-providers libibverbs1 librdmacm1 \
    && DRIVER_PACKAGES=$(apt list --installed 2>/dev/null | grep libnvidia- | cut -d/ -f1) \
    && echo "-- Removing files from driver packages brought by ucx-cuda which should not be in the container:" ${DRIVER_PACKAGES} \
    && for pkg in ${DRIVER_PACKAGES}; do \
        dpkg -L ${pkg} | xargs -I {} sh -c 'if [ ! -d "{}" ]; then rm -v "{}"; fi'; \
    done

# libvulkan-dev depends on python3 package which pulls python3.10. re-override with python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# PIP INSTALLS
#  requirements.dev.txt
#    coverage - test coverage of python tests
#    pytest*  - testing
#    pillow - convert_gxf_entities_to_images dependency
#  requirements
#    pip - 22.0.2+ needed for PEP 600 and a version that is greater than the Ubuntu 22.04's python3-pip package for python wheel symlinks resolution
#    cupy-cuda - dependency for holoscan python + examples
#    cloudpickle - dependency for distributed apps
COPY python/requirements.dev.txt /tmp/requirements.dev.txt
COPY python/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    python3 -m pip install -r /tmp/requirements.dev.txt -r /tmp/requirements.txt

# A pre-existing NumPy 1.x may have been kept by the above requirements.txt. Explicitly install 2.x
RUN --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    python3 -m pip install "numpy>2.0"

# Disable the keep-archives setting at the end of the build stage
# so it doesn't persist for users interacting directly with this image.
RUN KEEP_APT_ARCHIVES_CONF="/etc/apt/apt.conf.d/99-keep-archives"; \
    if [ -f "${KEEP_APT_ARCHIVES_CONF}" ]; then \
        mv "${KEEP_APT_ARCHIVES_CONF}" "${KEEP_APT_ARCHIVES_CONF}.disabled"; \
        echo "Disabled ${KEEP_APT_ARCHIVES_CONF} at the end of the build stage."; \
    fi

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
COPY --from=onnxruntime ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${ONNX_RUNTIME}"

# Copy Libtorch
ARG LIBTORCH_VERSION
ENV LIBTORCH=/opt/libtorch/${LIBTORCH_VERSION}
COPY --from=libtorch-downloader ${LIBTORCH} ${LIBTORCH}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${LIBTORCH}"

# Copy gRPC
ARG GRPC_VERSION
ENV GRPC=/opt/grpc/${GRPC_VERSION}
COPY --from=grpc-builder ${GRPC} ${GRPC}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GRPC}"

# Copy GXF
ENV GXF=/opt/nvidia/gxf/gxf-install
COPY --from=gxf-downloader ${GXF} ${GXF}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GXF}"
ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/gxf/${GXF_VERSION}/python"

############################################################################################
# dGPU specific build stage
############################################################################################
FROM build-generic AS build-dgpu

# no-op

############################################################################################
# iGPU specific build stage
############################################################################################
FROM build-generic AS build-igpu

# The iGPU CMake build depends on libnvcudla.so as well as libnvdla_compiler.so, which are
# part of the L4T BSP. As such, they should not be in the container, but mounted at runtime
# (which the nvidia container runtime handles). However, we need the symbols at build time
# for the TensorRT libraries to resolve. Since there is no stub library (unlike libcuda.so),
# we need to include them in our builder. We use a separate stage so that `run build` can
# use it if needed, but `run launch` (used to run apps in the container) doesn't need to.
WORKDIR /opt/nvidia/dla
RUN --mount=type=bind,from=dla-downloader,source=/opt/nvidia/dla,target=/dla \
    if [ ${GPU_TYPE} = "igpu" ]; then \
        dpkg -x /dla/l4t_core.deb / \
        && dpkg -x /dla/l4t_cuda.deb / \
        && dpkg -x /dla/l4t_dla.deb /; \
    fi

############################################################################################
# Final build stage (based on GPU type)
############################################################################################
FROM build-${GPU_TYPE} AS build
