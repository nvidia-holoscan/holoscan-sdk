# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG ONNX_RUNTIME_VERSION=1.24.2
ARG ONNX_RUNTIME_STRATEGY=downloader # or builder
ARG PYTORCH_IGPU_VERSION=2.11.0
ARG PYTORCH_DGPU_VERSION=2.11.0
ARG NCCL_VERSION=2.29  # match DLFW 26.03 development stack
ARG LIBCUSPARSELT_CU12_VERSION=0.8  # strict compat to match PyTorch versions
ARG LIBCUSPARSELT_CU13_VERSION=0.9      # match DLFW 26.03 development stack
ARG GRPC_VERSION=1.54.2
ARG GXF_CU12_VERSION=5.7.0_20260515_6a50c8f1a_holoscan-sdk-cu12
ARG GXF_CU13_VERSION=5.7.0_20260515_6a50c8f1a_holoscan-sdk-cu13
ARG DOCA_VERSION=3.3.0
ARG TENSORRT_CU12_VERSION=10.3  # TRT 10.3 is the last version that supports CUDA 12 on sbsa 22.04
ARG TENSORRT_CU13_VERSION=10.16
ARG UCX_VERSION=1.20.0
ARG GDRCOPY_VERSION=2.5.1  # MIT license - bundled with UCX for GPU Direct RDMA support
ARG NSYS_VERSION=2025.3.1  # at least 2025.3 required for CUDA 13.0 support
ARG OPENSSL_VERSION=3.0.19  # Use latest LTS source version with CVE fixes
ARG YAML_CPP_VERSION=0.8.0
ARG FASTDDS_VERSION=3.4.2
ARG FASTDDS_GEN_VERSION=4.3.0
ARG NVCOMP_VERSION=5.0.0.6    # Pin to <5.1 for stable CRC32 functionality on arm64 platforms.

############################################################
# Generic base image
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
FROM ${BASE_IMAGE} AS base-generic

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

############################################################
# iGPU base
############################################################
FROM base-generic AS base-igpu

# Install L4T APT repo on iGPU base
RUN L4T_APT_REPO_URL="https://repo.download.nvidia.com/jetson"; \
    L4T_APT_REPO_KEY="$L4T_APT_REPO_URL/jetson-ota-public.asc"; \
    L4T_APT_REPO_FILE="/usr/share/keyrings/jetson-ota-archive-keyring.gpg"; \
    L4T_APT_SOURCE_FILE="/etc/apt/sources.list.d/nvidia-l4t-apt-source.list"; \
    L4T_APT_BRANCH="r36.4"; \
    curl -sSL "$L4T_APT_REPO_KEY" | gpg --dearmor -o "$L4T_APT_REPO_FILE"; \
    echo "deb [signed-by=$L4T_APT_REPO_FILE] $L4T_APT_REPO_URL/common/ $L4T_APT_BRANCH main" > "$L4T_APT_SOURCE_FILE"

############################################################
# dGPU base
############################################################
FROM base-generic AS base-dgpu

# Set up CUDA APT repo on dGPU base if missing
RUN if ! grep -q "developer.download.nvidia.com/compute/cuda/repos" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then \
        OS_CODENAME=$(source /etc/os-release && echo "ubuntu${VERSION_ID//./}"); \
        CUDA_PLATFORM=$(uname -m); \
        if [ ${CUDA_PLATFORM} = "aarch64" ]; then \
            CUDA_PLATFORM="sbsa"; \
        fi; \
        CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_CODENAME}/${CUDA_PLATFORM}"; \
        CUDA_KEYRING="/usr/share/keyrings/cuda-archive-keyring.gpg"; \
        curl -fsSL "${CUDA_REPO_URL}/cuda-archive-keyring.gpg" -o "${CUDA_KEYRING}"; \
        echo "deb [signed-by=${CUDA_KEYRING}] ${CUDA_REPO_URL} /" > /etc/apt/sources.list.d/cuda.list; \
    fi

############################################################
# Base
############################################################
FROM base-${GPU_TYPE} AS base

# no-op, just choosing based on gpu type

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
# Python base (dGPU)
############################################################
FROM base AS python-base-dgpu

# Ensure we use Python 3.12 for dGPU stack since the dev container (released on NGC)
# is currently based on DLFW containers which are now on ubuntu 24.04 / python 3.12.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && curl -fsSL --retry 5 --retry-delay 3 \
        "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu ${OS_CODENAME} main" \
        > /etc/apt/sources.list.d/deadsnakes-ppa.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-dev \
    && apt purge -y \
        python3-pip \
    && apt-get autoremove --purge -y

# Enforce python 3.12 as system python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

############################################################
# Python base (iGPU)
############################################################
FROM base AS python-base-igpu

# In this case we stick to system python (3.10 for JP6 or IGX OS 1.* for Orin iGPU)
# since we need python 3.10 for pytorch wheels on JP6/IGX OS 1.*.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        python3 \
        python3-dev \
    && apt purge -y \
        python3-pip \
    && apt-get autoremove --purge -y

############################################################
# Python base
############################################################
FROM python-base-${GPU_TYPE} AS python-base

# Ensure python is the same as python3
RUN update-alternatives --install /usr/bin/python python $(command -v python3) 1

# Allow future pip installs on the system, no need for a
# venv in our container.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Get recent pip from pypa.io
# Will work whether using system python or deadsnakes' python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

############################################################
# Build tools
############################################################
FROM python-base AS build-tools

# Install build tools
# Note: Pin to CMake 3.31 to work around 3.31.12 error: "cmake : Breaks: cmake-data (< 4.3)"
#       CMake Issue#27775
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
    && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
        | gpg --dearmor -o "$KW_KEYRING" \
    && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake="3.31.11*" \
        cmake-data="3.31.11*" \
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
            rm -f "${file}"
            printf "${DELETED_LIB_MSG}" "${lib_size_human}" "${pkg}" > "${file}.deleted.txt"
        fi
    done
done
EOF

#  nvcc: needed by holoviz, holoinfer (cuda kernels), cmake (find CUDAToolkit)
#  cudart-dev: needed by holoscan core
#  nvrtc-dev: needed by holoscan core, and cupy (runtime only)
#  nvml-dev: needed by system_monitor (GPU metrics)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter="-") \
    && apt-get install --no-install-recommends -y \
        cuda-nvcc-${CUDA_MAJOR_MINOR} \
        cuda-cudart-dev-${CUDA_MAJOR_MINOR} \
        cuda-nvrtc-dev-${CUDA_MAJOR_MINOR} \
        cuda-nvml-dev-${CUDA_MAJOR_MINOR} \
    && echo "-- Deleting unused static libs:" \
    && packages=$(dpkg -l | grep cuda-nvrtc | awk '{print $2}') \
    && static_libs=$(dpkg -L $packages | grep '\.a$' || true) \
    && cleanup_unwanted_libs $static_libs

############################################################
# nsight-systems-cli
############################################################
FROM cuda-dev AS nsight-cli-dev

# The cuda-nsight-systems-${CUDA_MAJOR_MINOR} package is large as it also includes nsys-ui.
# We can install the lighter weight nsight-systems-cli instead, but this requires adding the
# NVIDIA DevTools APT repo.
ARG NSYS_VERSION

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    UBUNTU_RELEASE=$(source /etc/lsb-release && echo "$DISTRIB_RELEASE" | tr -d .) \
    && ARCH_STRING=$(dpkg --print-architecture) \
    && DEVTOOLS_REPO_URL="https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_RELEASE}/${ARCH_STRING}" \
    && echo "DEVTOOLS_REPO_URL=${DEVTOOLS_REPO_URL}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends gnupg \
    && curl -fsSL "${DEVTOOLS_REPO_URL}/nvidia.pub" | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] ${DEVTOOLS_REPO_URL} /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nsight-systems-cli-${NSYS_VERSION}

############################################################
# CUDA Toolkit dev
############################################################
FROM nsight-cli-dev AS cuda-toolkit-dev

# Install additional CUDA apt dependencies
#  cuda-cupti: needed by libtorch
#  cuda-nvtx: needed by libtorch
#  libcublas: runtime needed by libtorch, onnxruntime, cupy-cuda, and headers (-dev) for matx.
#  libcufft: runtime needed by libtorch, onnxruntime, cupy, and headers (-dev) for matx.
#  libcufile: needed by libtorch.
#  libcurand: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libcusolver: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libcusparse: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libnpp-dev: needed by bayer_demosaic, format_converter.
#  libnvjitlink: needed by libtorch, cupy-cuda.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter="-") \
    && apt-get install -y --no-install-recommends \
        cuda-cupti-${CUDA_MAJOR_MINOR} \
        cuda-nvtx-${CUDA_MAJOR_MINOR} \
        libcublas-dev-${CUDA_MAJOR_MINOR} \
        libcufft-dev-${CUDA_MAJOR_MINOR} \
        libcufile-${CUDA_MAJOR_MINOR} \
        libcurand-dev-${CUDA_MAJOR_MINOR} \
        libcusolver-dev-${CUDA_MAJOR_MINOR} \
        libcusparse-dev-${CUDA_MAJOR_MINOR} \
        libnpp-dev-${CUDA_MAJOR_MINOR} \
        libnvjitlink-${CUDA_MAJOR_MINOR} \
    && echo "-- Deleting unused static libs:" \
    && packages=$(dpkg -l | grep \
        -e cuda-cupti \
        -e cuda-nvtx \
        -e libcublas \
        -e libcufft \
        -e libcufile \
        -e libcurand \
        -e libcusolver \
        -e libcusparse \
        -e libnpp \
        -e libnvjitlink \
        | awk '{print $2}') \
    && static_libs=$(dpkg -L $packages | grep '\.a$' || true) \
    && cleanup_unwanted_libs $static_libs

############################################################
# Cudnn dev
############################################################
FROM cuda-toolkit-dev AS cudnn-dev

# Install CUDNN (needed by libtorch)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libcudnn9-cuda-${CUDA_MAJOR} \
    && echo "-- Deleting large unused libs:" \
    && SYSTEM_LIBS_ROOT="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)" \
    && cleanup_unwanted_libs \
        "${SYSTEM_LIBS_ROOT}/libcudnn_cnn.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_engines_precompiled.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_engines_runtime_compiled.*" \
        "${SYSTEM_LIBS_ROOT}/libcudnn_heuristic.*"

############################################################
# TensorRT dev
############################################################
FROM cudnn-dev AS tensorrt-dev

ARG TENSORRT_CU12_VERSION
ARG TENSORRT_CU13_VERSION
# libnvinfer-safe-headers-dev is a transitive dep of libnvinfer-dev at TRT 10.16+; the pin blocks
# apt from selecting the newer TRT 11 candidate as the resolved dep. Not shipped for TRT 10.3
# (holoscan's cu12 pin, see TENSORRT_CU12_VERSION) or in the Jetson L4T repo, so feature-detect.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && TRT_VERSION_VAR_NAME="TENSORRT_CU${CUDA_MAJOR}_VERSION" \
    && TRT_VERSION=$(apt-cache madison libnvinfer10 | grep "${!TRT_VERSION_VAR_NAME}" | grep "+cuda${CUDA_MAJOR}" | head -n 1 | awk '{print $3}') \
    && EXTRA_TRT_PKGS="" \
    && if apt-cache madison libnvinfer-safe-headers-dev 2>/dev/null | grep -Fq "| ${TRT_VERSION} |"; then \
           EXTRA_TRT_PKGS="libnvinfer-safe-headers-dev=${TRT_VERSION}"; \
       fi \
    && apt-get install -y --no-install-recommends \
        libnvonnxparsers-dev="${TRT_VERSION}" \
        libnvonnxparsers10="${TRT_VERSION}" \
        libnvinfer-plugin-dev="${TRT_VERSION}" \
        libnvinfer-headers-plugin-dev="${TRT_VERSION}" \
        libnvinfer-plugin10="${TRT_VERSION}" \
        libnvinfer-dev="${TRT_VERSION}" \
        libnvinfer-headers-dev="${TRT_VERSION}" \
        libnvinfer10="${TRT_VERSION}" \
        ${EXTRA_TRT_PKGS} \
    && echo "-- Deleting unused static libs:" \
    && packages=$(dpkg -l | grep -e nvinfer -e nvonnxparsers | awk '{print $2}') \
    && static_libs=$(dpkg -L $packages | grep '\.a$' || true) \
    && cleanup_unwanted_libs $static_libs \
    && echo "-- Deleting large unused libs:" \
    && SYSTEM_LIBS_ROOT="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)" \
    && cleanup_unwanted_libs \
        "${SYSTEM_LIBS_ROOT}/libnvinfer_builder_resource_win.*"

############################################################################################
# dGPU specific inference dependencies
############################################################################################
FROM tensorrt-dev AS infer-dev-dgpu

ARG NCCL_VERSION
ARG LIBCUSPARSELT_CU12_VERSION
ARG LIBCUSPARSELT_CU13_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    if [ "${CUDA_MAJOR}" = "13" ]; then \
        LIBCUSPARSELT_VERSION="${LIBCUSPARSELT_CU13_VERSION}"; \
    else \
        LIBCUSPARSELT_VERSION="${LIBCUSPARSELT_CU12_VERSION}"; \
    fi; \
    apt-get update \
    # Get the exact NCCL package version for the specified NCCL_VERSION and CUDA_MAJOR with a single grep call.
    && NCCL_APT_VERSION=$(apt-cache madison libnccl2 | grep -o "${NCCL_VERSION}[^ ]*+cuda${CUDA_MAJOR}[^ ]*" | head -n 1) \
    && if [ $(uname -m) = "aarch64" ]; then \
        CONDITIONAL_LIBS="nvpl-blas nvpl-lapack"; \
    fi \
    && apt-get purge -y libcusparselt0 libcusparselt-dev \
    && apt-get autoremove --purge -y \
    && apt-get install -y --no-install-recommends \
        libnccl2=${NCCL_APT_VERSION} \
        libnvshmem3-cuda-${CUDA_MAJOR} \
        libcusparselt0-cuda-${CUDA_MAJOR}=${LIBCUSPARSELT_VERSION}* \
        ${CONDITIONAL_LIBS}

############################################################################################
# iGPU specific inference dependencies
############################################################################################
FROM tensorrt-dev AS infer-dev-igpu

# The iGPU CMake build depends on libnvcudla.so as well as libnvdla_compiler.so, which are
# part of the L4T BSP. As such, they should not be in the container, but mounted at runtime
# (which the nvidia container runtime handles). However, we need the symbols at build time
# for the TensorRT libraries to resolve. Since there is no stub library (unlike libcuda.so),
# we need to include them in our builder. Consider moving this to an optional stage if you
# plan to use this container for runtime in a portable fashion across JP6 versions in case
# the DLA symbols are not ABI compatible across versions.
WORKDIR /opt/nvidia/dla
RUN --mount=type=bind,from=dla-downloader,source=/opt/nvidia/dla,target=/dla \
    dpkg -x /dla/l4t_core.deb / \
    && dpkg -x /dla/l4t_cuda.deb / \
    && dpkg -x /dla/l4t_dla.deb /

# cudss could be needed by torch for Jetpack 6.
# It is missing from the L4T 36.4 repo at this time.
ARG CUDSS_VERSION=0.6.0.5
RUN cudss_deb_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcudss0-cuda-${CUDA_MAJOR}_${CUDSS_VERSION}-1_arm64.deb" \
    && echo "cudss_deb_url: ${cudss_deb_url}" \
    && curl -fSL# -o libcudss.deb ${cudss_deb_url} \
    && apt-get install --no-install-recommends -y ./libcudss.deb \
    && rm libcudss.deb

# libcusolver 11.7.1.2 is needed by torch for Jetpack 6 (symbol: cusolverDnXsyevBatched_bufferSize)
# Latest from the L4T 36.4 repo is 11.6.4.69
ARG LIBCUSOLVER_VERSION=11.7.1.2
RUN CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 | tr . -) \
    && cusolver_deb_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcusolver-${CUDA_MAJOR_MINOR}_${LIBCUSOLVER_VERSION}-1_arm64.deb" \
    && echo "cusolver_deb_url: ${cusolver_deb_url}" \
    && curl -fSL# -o libcusolver.deb ${cusolver_deb_url} \
    && apt-get install --no-install-recommends -y ./libcusolver.deb \
    && rm libcusolver.deb

# - libnuma1 needed by cudss
# - libopenblas0 needed by torch
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libnuma1 \
        libopenblas0

############################################################################################
# Final inference dependencies
############################################################################################
FROM infer-dev-${GPU_TYPE} AS infer-dev

# Ensure cuda dependencies with non-standard install paths are found by the dynamic linker.
# - cusparselt
# - nvshmem
# - cudss
RUN for pkg in "libcusparselt0-cuda-${CUDA_MAJOR}" "libnvshmem3-cuda-${CUDA_MAJOR}" "libcudss0-cuda-${CUDA_MAJOR}"; do \
        if dpkg -s "${pkg}" >/dev/null 2>&1; then \
            dpkg -L "${pkg}" | grep -E ".so[\\.0-9]+" | xargs dirname | uniq | tee /etc/ld.so.conf.d/${pkg}.conf; \
            ldconfig; \
        fi; \
    done

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
# sccache
############################################################
FROM base AS sccache-downloader

WORKDIR /opt/sccache

# Set sccache version
ENV SCCACHE_VERSION=v0.14.0-rapids.1
ENV SCCACHE_BASE_URL=https://github.com/rapidsai/sccache/releases/download

# Download and extract the binary
RUN curl -S -# -L -o sccache.tar.gz \
    ${SCCACHE_BASE_URL}/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl.tar.gz && \
    tar -xzf sccache.tar.gz -C /opt/sccache --strip-components=1 sccache-${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl/sccache

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

############################################################
# ONNX Runtime (Build)
############################################################
FROM tensorrt-dev AS onnxruntime-builder
ARG ORT_DIR=/opt/onnxruntime
ARG ONNX_RUNTIME_VERSION

# Need cudnn.h for ONNX Runtime 1.24 build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt install -y --no-install-recommends libcudnn9-dev-cuda-${CUDA_MAJOR}

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

# Expected SHA256 checksums for ORT tarballs (see public/checksums/)
COPY checksums/onnxruntime-${ONNX_RUNTIME_VERSION}.sha256 /tmp/onnxruntime.sha256

# Download ORT binaries from artifactory
WORKDIR /opt/onnxruntime
RUN CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2) \
    && if [ ${CUDA_MAJOR} = "13" ]; then \
        # Use ONNX Runtime CUDA 13.0 pre-built binary for convenience
        CUDA_MAJOR_MINOR="13.0"; \
    fi \
    && ORT_ARCHIVE="onnxruntime-${ONNX_RUNTIME_VERSION}-cuda-${CUDA_MAJOR_MINOR}-$(uname -m).tar.gz" \
    && ORT_URL="https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/onnxruntime/${ORT_ARCHIVE}" \
    && curl -S -L -# -D /tmp/ort.headers -o ort.tgz "${ORT_URL}" \
    && echo "Downloaded ONNX Runtime from ${ORT_ARCHIVE}" \
    && ARCHIVE_SIZE=$(stat -c%s "ort.tgz") \
    && if [ "${ARCHIVE_SIZE}" -lt 1024 ]; then \
        echo "Error: Downloaded archive ort.tgz is less than 1KB in size (${ARCHIVE_SIZE} bytes). Possible download error." >&2; \
        exit 2; \
    fi \
    && EXPECTED_SHA=$(grep -F " ${ORT_ARCHIVE}" /tmp/onnxruntime.sha256 | awk '{print $1}') \
    && if [ -z "${EXPECTED_SHA}" ]; then \
        echo "Error: No SHA256 checksum in repository for ${ORT_ARCHIVE}" >&2; \
        exit 2; \
    fi \
    && HEADER_SHA=$(grep -i '^x-checksum-sha256:' /tmp/ort.headers | awk '{print $2}' | tr -d '\r') \
    && if [ "${HEADER_SHA}" != "${EXPECTED_SHA}" ]; then \
        echo "Error: Repository SHA256 does not match Artifactory X-Checksum-Sha256 for ${ORT_ARCHIVE}" >&2; \
        echo "  Repository:  ${EXPECTED_SHA}" >&2; \
        echo "  Artifactory: ${HEADER_SHA}" >&2; \
        exit 2; \
    fi \
    && echo "${EXPECTED_SHA} ort.tgz" | sha256sum -c -
RUN mkdir -p ${ONNX_RUNTIME_VERSION}
RUN tar -xf ort.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 2 --no-same-owner --no-same-permissions

############################################################
# ONNX Runtime (Build or Download)
############################################################
FROM onnxruntime-${ONNX_RUNTIME_STRATEGY} AS onnxruntime

############################################################
# PyTorch (base)
############################################################
FROM python-base AS pytorch-downloader-base
ARG TORCH_WHL_DIR=/opt/wheels

############################################################
# PyTorch (dGPU downloader)
############################################################
FROM pytorch-downloader-base AS pytorch-downloader-dgpu
ARG PYTORCH_DGPU_VERSION

# Install torch wheel
RUN --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter=""); \
    if [ ${CUDA_MAJOR} = "13" ]; then \
        # PyTorch CUDA 13.2 support is scheduled for PyTorch 2.12.0.
        # For now, use major-compatible CUDA 13.0 wheels.
        CUDA_MAJOR_MINOR="130"; \
    fi; \
    PYTORCH_VERSION="${PYTORCH_DGPU_VERSION}+cu${CUDA_MAJOR_MINOR}"; \
    INDEX_URL="https://download.pytorch.org/whl"; \
    python3 -m pip download \
        --dest ${TORCH_WHL_DIR} \
        --pre \
        --no-deps \
        --index-url ${INDEX_URL} \
        torch=="${PYTORCH_VERSION}"

############################################################
# PyTorch (iGPU downloader)
############################################################
FROM pytorch-downloader-base AS pytorch-downloader-igpu
ARG PYTORCH_IGPU_VERSION

RUN python3 -m pip download \
        --dest ${TORCH_WHL_DIR} \
        --no-deps \
        --index-url "https://pypi.jetson-ai-lab.io/jp6/cu126" \
        torch=="${PYTORCH_IGPU_VERSION}"

############################################################
# PyTorch (common downloader)
############################################################
FROM pytorch-downloader-${GPU_TYPE} AS pytorch-downloader

# no-op, just choosing based on gpu type

############################################################
# PyTorch (install)
############################################################
FROM infer-dev AS pytorch-dev

# Install PyTorch
# --no-index and --find-links to pull from the local download only.
# --no-deps to skip cuda/nvidia wheel dependencies already present on system for c++ use.
# cleanup_unwanted_libs to remove revendored cuda/nvidia libs (old packaging strategy for pytorch).
RUN --mount=type=bind,from=pytorch-downloader,source=/opt/wheels,target=/opt/wheels \
    --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    python3 -m pip install torch --no-index --no-deps --find-links=/opt/wheels \
    && echo "-- Deleting revendored cuda libs that could conflict with system cuda libs:" \
    && TORCH_LIB_DIR=$(python3 -m pip show torch | grep Location | awk '{print $2}')/torch/lib \
    && cleanup_unwanted_libs \
        "${TORCH_LIB_DIR}/libcu*" \
        "${TORCH_LIB_DIR}/libnv*" \
        "${TORCH_LIB_DIR}/libnccl*"

# Show torch install info
RUN python3 -m pip show torch

# Patch Torch CMake config to remove kineto warning.
RUN TORCH_SITE_DIR=$(python3 -m pip show torch | grep Location | awk '{print $2}') && \
    find ${TORCH_SITE_DIR} -type f -name "*Config.cmake" -exec sed -i '/kineto/d' {} +

# Install pip dependencies for PyTorch apart from:
# - nvidia-* (already on system, see above)
# - triton (not needed)
# Note: consider switching to uv.pip.no-emit-package instead when supporting uv.
RUN --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    python3 - <<'PY' | python3 -m pip install -r /dev/stdin
from importlib.metadata import metadata
reqs = metadata("torch").get_all("Requires-Dist") or []
for req in reqs:
    if not req.lower().startswith(("nvidia-", "triton")):
        print(req)
PY

# Sanity checks that torch can be imported + cuda related info
RUN python3 -- <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
# Temporary monkey patch to get cuda compile info without a GPU during build.
torch.cuda.is_available = lambda: True
print(f"PyTorch CUDA arch list: {torch.cuda.get_arch_list()}")
print(f"PyTorch CUDA gencode flags: {torch.cuda.get_gencode_flags()}")
print(torch.__config__.show())
PY

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
# yaml-cpp libraries
############################################################
FROM build-tools AS yaml-cpp-builder
ARG YAML_CPP_VERSION
ARG MAX_PROC

WORKDIR /opt/yaml-cpp
RUN git clone --depth 1 --branch ${YAML_CPP_VERSION} \
        https://github.com/jbeder/yaml-cpp.git src
# Apply patch for AddressSanitizer (push_back no_sanitize_address)
COPY patches/yaml-cpp.patch /opt/yaml-cpp/yaml-cpp.patch
RUN cd src && git apply /opt/yaml-cpp/yaml-cpp.patch
WORKDIR /opt/yaml-cpp/src
# Do not use -fvisibility=hidden: the static lib is linked into shared libs (e.g.
# libholoscan.so), and hidden visibility causes "relocation against undefined
# hidden symbol" when resolving std::shared_ptr and other libstdc++ symbols.
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/yaml-cpp/${YAML_CPP_VERSION} \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DYAML_CPP_BUILD_CONTRIB=OFF \
        -DYAML_CPP_BUILD_TOOLS=OFF \
        -DYAML_BUILD_SHARED_LIBS=OFF \
        -DYAML_CPP_INSTALL=ON
RUN cmake --build build -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN cmake --install build --prefix /opt/yaml-cpp/${YAML_CPP_VERSION}

############################################################
# Fast DDS (eProsima) - for IPC / DDS-based communication
############################################################
# We build from source instead of using eProsima's .deb package because:
# - There is no public apt repository; the .deb must be downloaded manually from
#   https://eprosima.com/index.php/downloads-all (no stable URL for automation).
# - Official binaries are documented for amd64 only; this image supports TARGETARCH
#   (amd64 and arm64), and building from source guarantees Fast DDS for both.
# - Building installs to /opt/fastdds/${FASTDDS_VERSION}, which downstream stages
#   expect (CMAKE_PREFIX_PATH, ldconfig). Package install would use system paths.
# See: https://fast-dds.docs.eprosima.com/en/3.4.x/installation/binaries/binaries_linux.html
#
ARG FASTDDS_VERSION=3.4.2
FROM build-tools AS fastdds-builder
ARG TARGETARCH
ARG GPU_TYPE
ARG FASTDDS_VERSION=3.4.2
ARG MAX_PROC

# Fast DDS build dependencies (see https://fast-dds.docs.eprosima.com/en/3.4.x/installation/sources/sources_linux.html)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libasio-dev \
        libtinyxml2-dev \
        libssl-dev

ENV FASTDDS_INSTALL_DIR=/opt/fastdds/${FASTDDS_VERSION}

# 1. foonathan_memory_vendor (v1.3.2 for Fast DDS 3.4.2)
WORKDIR /opt/fastdds/build
RUN git clone --depth 1 --branch v1.3.2 https://github.com/eProsima/foonathan_memory_vendor.git foonathan_memory_vendor
RUN cmake -S foonathan_memory_vendor -B foonathan_memory_vendor/build -G Ninja \
    -D CMAKE_INSTALL_PREFIX=${FASTDDS_INSTALL_DIR} \
    -D BUILD_SHARED_LIBS=ON
RUN cmake --build foonathan_memory_vendor/build -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN cmake --install foonathan_memory_vendor/build --prefix ${FASTDDS_INSTALL_DIR}

# 2. Fast-CDR (v2.3.5 for Fast DDS 3.4.2)
RUN git clone --depth 1 --branch v2.3.5 https://github.com/eProsima/Fast-CDR.git Fast-CDR
RUN cmake -S Fast-CDR -B Fast-CDR/build -G Ninja \
    -D CMAKE_INSTALL_PREFIX=${FASTDDS_INSTALL_DIR}
RUN cmake --build Fast-CDR/build -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN cmake --install Fast-CDR/build --prefix ${FASTDDS_INSTALL_DIR}

# 3. Fast DDS (v3.4.2)
RUN git clone --depth 1 --branch v${FASTDDS_VERSION} https://github.com/eProsima/Fast-DDS.git Fast-DDS
RUN cmake -S Fast-DDS -B Fast-DDS/build -G Ninja \
    -D CMAKE_INSTALL_PREFIX=${FASTDDS_INSTALL_DIR} \
    -D CMAKE_PREFIX_PATH=${FASTDDS_INSTALL_DIR} \
    -D CMAKE_BUILD_TYPE=Release
RUN cmake --build Fast-DDS/build -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN cmake --install Fast-DDS/build --prefix ${FASTDDS_INSTALL_DIR}

############################################################
# Fast-DDS-Gen (eProsima) - IDL compiler for Fast DDS (Java / Gradle)
############################################################
# Not bundled with Fast-DDS; needed to regenerate holo_ipc / example Fast DDS types from .idl.
# Install layout: /opt/fastdds-gen/${FASTDDS_GEN_VERSION}/{fastddsgen.jar,bin/fastddsgen}
# See: https://github.com/eProsima/Fast-DDS-Gen
#
FROM build-tools AS fastdds-gen-builder
ARG FASTDDS_GEN_VERSION=4.3.0
ARG MAX_PROC
ARG TARGETARCH
ARG GPU_TYPE
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV FAST_DDS_GEN_INSTALL_DIR=/opt/fastdds-gen/${FASTDDS_GEN_VERSION}
WORKDIR /opt/fastdds-gen/build
RUN git clone --depth 1 --branch v${FASTDDS_GEN_VERSION} \
    https://github.com/eProsima/Fast-DDS-Gen.git Fast-DDS-Gen
WORKDIR /opt/fastdds-gen/build/Fast-DDS-Gen
RUN ./gradlew assemble --no-daemon -Dorg.gradle.parallel=true \
    -Dorg.gradle.workers.max=$(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN mkdir -p "${FAST_DDS_GEN_INSTALL_DIR}/bin" \
    && cp build/libs/fastddsgen.jar "${FAST_DDS_GEN_INSTALL_DIR}/" \
    && printf '#!/bin/sh\nexec java -jar "%s/fastddsgen.jar" "$@"\n' "${FAST_DDS_GEN_INSTALL_DIR}" \
        > "${FAST_DDS_GEN_INSTALL_DIR}/bin/fastddsgen" \
    && chmod +x "${FAST_DDS_GEN_INSTALL_DIR}/bin/fastddsgen"


############################################################
# GXF
############################################################
FROM base AS gxf-downloader
ARG GXF_CU12_VERSION
ARG GXF_CU13_VERSION
ARG CUDA_MAJOR

WORKDIR /tmp/gxf
RUN set -x; \
    if [ "${CUDA_MAJOR}" = "13" ]; then \
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

WORKDIR /opt/nvidia/gxf
RUN tar xzf /tmp/gxf/gxf.tgz --strip-components 2 --no-same-owner --no-same-permissions

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
        > /etc/apt/preferences.d/doca-pin \
    && DOCA_LATEST_URL="${DOCA_REPO_ROOT}/latest/${DISTRO}/${DOCA_ARCH}/" \
    && DOCA_LATEST_GPG_KEY_PATH="/etc/apt/trusted.gpg.d/doca-latest-keyring.gpg" \
    && curl -fsSL ${DOCA_LATEST_URL}/doca_keyring.gpg -o ${DOCA_LATEST_GPG_KEY_PATH} \
    && echo "deb [signed-by=${DOCA_LATEST_GPG_KEY_PATH}] ${DOCA_LATEST_URL} ./" \
        >> /etc/apt/sources.list.d/doca.list

# Setup LLVM APT repository (for newer clang-tidy)
# https://apt.llvm.org/
ARG LLVM_VERSION=18
RUN DISTRO_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] https://apt.llvm.org/${DISTRO_CODENAME}/ llvm-toolchain-${DISTRO_CODENAME}-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list

############################################################
# UCX (with RDMA and gdrcopy support)
############################################################
FROM cuda-dev AS ucx-builder

# Setup apt repositories (use DOCA repo for RDMA packages)
COPY --from=apt-repo-config /etc/apt/keyrings/ /etc/apt/keyrings/
COPY --from=apt-repo-config /etc/apt/sources.list.d/ /etc/apt/sources.list.d/
COPY --from=apt-repo-config /etc/apt/preferences.d/ /etc/apt/preferences.d/
COPY --from=apt-repo-config /etc/apt/trusted.gpg.d/ /etc/apt/trusted.gpg.d/

WORKDIR /opt/ucx
ARG UCX_VERSION
ARG GDRCOPY_VERSION
ARG MAX_PROC

# Install build tools and RDMA dependencies for UCX:
# Build tools: autoconf, automake
# RDMA dependencies (see https://openucx.readthedocs.io/en/master/faq.html):
#   - rdma-core: Core RDMA userspace libraries and utilities
#   - libibverbs-dev: Required for --with-verbs and --with-mlx5 (libuct_ib.so, libuct_ib_mlx5.so)
#   - librdmacm-dev: Required for --with-rdmacm (libuct_rdmacm.so)
#
# Note: xpmem and fuse plugins are not included in this release.
#       They will be included in a future release.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        libtool \
        rdma-core \
        libibverbs-dev \
        librdmacm-dev

# Build gdrcopy (MIT license) for GPU Direct RDMA support (libuct_cuda_gdrcopy.so)
# Note: The kernel module must be installed on the HOST separately.
#       This only builds the userspace library.
WORKDIR /opt/gdrcopy
RUN git clone --depth 1 --branch v${GDRCOPY_VERSION} \
        https://github.com/NVIDIA/gdrcopy.git src
WORKDIR /opt/gdrcopy/src
RUN make CUDA=/usr/local/cuda lib lib_install prefix=/opt/gdrcopy/install

WORKDIR /opt/ucx
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        --branch v${UCX_VERSION} \
        https://github.com/openucx/ucx.git src

# Build UCX with RDMA and gdrcopy support
# Plugins built:
#   - libuct_ib.so: InfiniBand transport
#   - libuct_ib_mlx5.so: Mellanox ConnectX optimized transport
#   - libuct_rdmacm.so: RDMA connection manager
#   - libuct_cuda_gdrcopy.so: GPU Direct RDMA copy (requires gdrcopy kernel module on host)
# Plugins NOT built (will be included in a future release):
#   - libuct_xpmem.so: Cross-process memory
#   - libucx_fuse.so: FUSE debugging interface
WORKDIR /opt/ucx/src
RUN ./autogen.sh && \
    ./contrib/configure-release --prefix=/opt/ucx/ \
    --enable-optimizations \
    --enable-mt \
    --enable-cma \
    --enable-devel-headers \
    --with-verbs \
    --with-mlx5 \
    --with-rdmacm \
    --with-cuda=/usr/local/cuda \
    --with-gdrcopy=/opt/gdrcopy/install \
    --without-knem \
    --without-xpmem \
    --without-fuse3 \
    --without-java \
    --without-go
RUN make -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN make install

# Copy gdrcopy library to UCX lib directory (required by libuct_cuda_gdrcopy.so at runtime)
RUN find /opt/gdrcopy/install/lib -name '*.so*' -exec cp -a {} /opt/ucx/lib/ \;

# UCX build configuration summary:
# - RDMA/InfiniBand support enabled (verbs, mlx5, rdmacm)
# - GPU Direct RDMA copy enabled (gdrcopy)
# - CUDA support enabled
# - Multi-threading enabled
# - xpmem and fuse3 will be included in a future release

############################################################
# Build image
############################################################
FROM pytorch-dev AS build

# Setup apt repositories
COPY --from=apt-repo-config /etc/apt/keyrings/ /etc/apt/keyrings/
COPY --from=apt-repo-config /etc/apt/sources.list.d/ /etc/apt/sources.list.d/
COPY --from=apt-repo-config /etc/apt/preferences.d/ /etc/apt/preferences.d/
COPY --from=apt-repo-config /etc/apt/trusted.gpg.d/ /etc/apt/trusted.gpg.d/

# APT INSTALLS
#  valgrind - dynamic analysis
#  clang-tidy-18 - static analysis (from LLVM apt repo, newer than Ubuntu's default v14)
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
#  doca-sdk-gpunetio: runtime library for HSB gpu_roce_transceiver operator
#  libdoca-sdk-gpunetio-dev: doca-gpunetio pkg-config + headers for HSB gpu_roce_transceiver operator
#    (requires DOCA "latest" repo — libdoca-sdk-gpunetio-dev was omitted from the pinned 3.3.0 directory)
#  libdoca-sdk-verbs-dev: provides doca-verbs pkg-config + headers for HSB gpu_roce_transceiver operator
ARG LLVM_VERSION=18
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind \
        clang-tidy-${LLVM_VERSION} \
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
        libtinyxml2-dev \
        ibverbs-providers libibverbs1 librdmacm1 \
        doca-sdk-gpunetio \
        libdoca-sdk-gpunetio-dev \
        libdoca-sdk-verbs-dev \
    && ln -sf /usr/bin/clang-tidy-${LLVM_VERSION} /usr/bin/clang-tidy

# Workaround: Remove cuda-compat-13-* which is known to not function properly.
# cuda-compat-13-0 does not provide display forward compatibility and does not
# meaningfully support CTK 13.x minor version forward compatibility, so we can
# safely remove it.
# Addresses build container runtime test failures on IGX Thor platforms with IGX-SW 2.0.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get purge -y cuda-compat-13-0 || true

# Installing libvulkan-dev may re-install the python3 package.
# Re-override to use a separate version if needed (see python-base-${GPU_TYPE} stages)
RUN if [ "${GPU_TYPE}" = "dgpu" ]; then \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1; \
    fi

# PIP INSTALLS
#  requirements.dev.txt
#    coverage - test coverage of python tests
#    pytest*  - testing
#  requirements
#    cloudpickle - dependency for distributed apps
#    cupy-cuda - dependency for holoscan python + examples
#    numpy - dependency for holoscan python + examples
#    pillow - convert_gxf_entities_to_images dependency
#    pip - 22.0.2+ needed for PEP 600 and a version that is greater than the Ubuntu 22.04's python3-pip package for python wheel symlinks resolution
COPY python/requirements.dev.txt /tmp/requirements.dev.txt
COPY python/requirements.cu${CUDA_MAJOR}.txt /tmp/requirements.cu${CUDA_MAJOR}.txt
# Note: --ignore-installed is required to handle a pip/apt conflict on Ubuntu 24.04.
# On Ubuntu 22.04, python3-yaml provides PyYAML 5.4.1 which doesn't satisfy >=6.0,
# so pip installs a fresh copy without issues.
# On Ubuntu 24.04, python3-yaml provides PyYAML 6.0.1 which satisfies >=6.0, but
# when --upgrade triggers an upgrade attempt, pip fails to uninstall the apt-installed
# package because it lacks a RECORD file (apt packages don't create these).
# The error is: "Cannot uninstall PyYAML 6.0.1 ... no RECORD file was found".
# Using --ignore-installed tells pip to install fresh copies without attempting to
# uninstall existing system packages.
RUN --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    python3 -m pip install --upgrade --ignore-installed \
        -r /tmp/requirements.dev.txt \
        -r /tmp/requirements.cu${CUDA_MAJOR}.txt

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

# Copy sccache
ENV SCCACHE=/opt/sccache
COPY --from=sccache-downloader ${SCCACHE}/sccache /usr/local/bin

# Copy ONNX Runtime
ARG ONNX_RUNTIME_VERSION
ENV ONNX_RUNTIME=/opt/onnxruntime/${ONNX_RUNTIME_VERSION}
COPY --from=onnxruntime ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${ONNX_RUNTIME}"

# Copy gRPC
ARG GRPC_VERSION
ENV GRPC=/opt/grpc/${GRPC_VERSION}
COPY --from=grpc-builder ${GRPC} ${GRPC}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GRPC}"

# Copy GXF
ENV GXF=/opt/nvidia/gxf
COPY --from=gxf-downloader ${GXF} ${GXF}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GXF}"

# Copy UCX
ENV UCX=/opt/ucx
COPY --from=ucx-builder ${UCX} ${UCX}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${UCX}"
RUN echo "/opt/ucx/lib" >> /etc/ld.so.conf.d/ucx.conf \
    && echo "/opt/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf \
    && ldconfig

# Copy yaml-cpp
ARG YAML_CPP_VERSION
ENV YAML_CPP=/opt/yaml-cpp/${YAML_CPP_VERSION}
COPY --from=yaml-cpp-builder ${YAML_CPP} ${YAML_CPP}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${YAML_CPP}"

# Copy Fast DDS (for IPC / holo_ipc)
ARG FASTDDS_VERSION=3.4.2
ENV FASTDDS=/opt/fastdds/${FASTDDS_VERSION}
COPY --from=fastdds-builder ${FASTDDS} ${FASTDDS}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${FASTDDS}"
RUN echo "${FASTDDS}/lib" >> /etc/ld.so.conf.d/fastdds.conf && ldconfig

# Fast-DDS-Gen (fastddsgen.jar + wrapper); JRE required at runtime to invoke java -jar
ARG FASTDDS_GEN_VERSION=4.3.0
ENV FAST_DDS_GEN=/opt/fastdds-gen/${FASTDDS_GEN_VERSION}
COPY --from=fastdds-gen-builder ${FAST_DDS_GEN} ${FAST_DDS_GEN}
ENV PATH="${PATH}:${FAST_DDS_GEN}/bin"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

############################################################################################
# GXF CMake build stage
############################################################################################
FROM build AS build-gxf-cmake

# Prevent existing GXF package from being visible in GXF build environment
RUN rm -rf /opt/nvidia/gxf

############################################################################################
# OpenSSL build stage (for Holoscan Sensor Bridge)
############################################################################################
FROM build-tools AS build-openssl

ARG OPENSSL_VERSION
WORKDIR /opt/openssl
RUN git clone --depth 1 --branch openssl-${OPENSSL_VERSION} \
    https://github.com/openssl/openssl.git src
WORKDIR /opt/openssl/src
# --openssldir: config/certs location (same as prefix). -Wl,-rpath,$(LIBRPATH): runtime
# library search path so the installed openssl binary finds libssl/libcrypto in /opt/openssl
# (NOTES-UNIX: non-default install locations require explicit rpath)
RUN ./config --prefix=/opt/openssl --openssldir=/opt/openssl \
    '-Wl,-rpath,$(LIBRPATH)'
ARG MAX_PROC
RUN make -j $(( `nproc` > ${MAX_PROC} ? ${MAX_PROC} : `nproc` ))
RUN make install

############################################################################################
# Holoscan Sensor Bridge CMake build stage
############################################################################################
FROM build AS build-holoscan-sensor-bridge-igpu
# No-op: Holoscan SDK + Hololink companion build currently supported for dGPU systems only.

FROM build AS build-holoscan-sensor-bridge-dgpu

# Install build dependencies for Holoscan Sensor Bridge
#  ibverbs* rdma*: needed for ConnectX RDMA support for Holoscan Sensor Bridge operators
#  libcurl-pp: Holoscan Sensor Bridge tools link dependency
#  OpenSSL: Holoscan Sensor Bridge tools link dependency
#  zlib: Holoscan Sensor Bridge emulator link dependency
#  iproute2: Provides "ip" network tools for Holoscan Sensor Bridge validation
#  net-tools: Provides "ifconfig" and other legacy network tools for Holoscan Sensor Bridge validation
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        ibverbs-providers libibverbs1 librdmacm1 \
        rdma-core \
        libibverbs-dev \
        librdmacm-dev \
        libcurlpp-dev \
        zlib1g-dev \
        net-tools \
        iproute2

# Install nvCOMP for CRC32 operator.
ARG NVCOMP_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holoscan-sdk-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holoscan-sdk-apt-lib-$TARGETARCH-$GPU_TYPE \
    if [ "${GPU_TYPE}" = "dgpu" ]; then \
        CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) \
        && apt-get update \
        && apt-get install -y \
            libnvcomp5-cuda-${CUDA_MAJOR}=${NVCOMP_VERSION}-1 \
            libnvcomp5-dev-cuda-${CUDA_MAJOR}=${NVCOMP_VERSION}-1 \
            libnvcomp5-static-cuda-${CUDA_MAJOR}=${NVCOMP_VERSION}-1 \
            nvcomp-cuda-${CUDA_MAJOR}=${NVCOMP_VERSION}-1 \
        && apt-mark hold \
            libnvcomp5-cuda-${CUDA_MAJOR} \
            libnvcomp5-dev-cuda-${CUDA_MAJOR} \
            libnvcomp5-static-cuda-${CUDA_MAJOR} \
            nvcomp-cuda-${CUDA_MAJOR} \
    ; fi

# Holoscan Sensor Bridge Python runtime dependencies
RUN --mount=type=bind,from=pytorch-downloader,source=/opt/wheels,target=/opt/wheels \
    --mount=type=cache,target=/root/.cache/pip,id=holoscan-sdk-pip-cache-$TARGETARCH-$GPU_TYPE \
    CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter=".") \
    && python3 -m pip install \
        cuda-python~=${CUDA_MAJOR_MINOR} \
        nvtx~=0.2.14

COPY --from=build-openssl /opt/openssl /opt/openssl
# Prepend OpenSSL so that the custom, latest build is found before the gRPC installation
ENV CMAKE_PREFIX_PATH="/opt/openssl:${CMAKE_PREFIX_PATH}"

############################################################
# Final stage
############################################################
FROM build-holoscan-sensor-bridge-${GPU_TYPE} AS final
