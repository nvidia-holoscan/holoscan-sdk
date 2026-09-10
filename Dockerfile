# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

############################################################
# Versions
############################################################
ARG PYTORCH_VERSION=2.11.0
ARG NCCL_VERSION=2.29  # match DLFW 26.03 development stack
ARG LIBCUSPARSELT_VERSION=0.9      # match DLFW 26.03 development stack
ARG TENSORRT_VERSION=10.16
ARG NSYS_VERSION=2026.4.1
ARG NVCOMP_VERSION=5.0.0.6    # Pin to <5.1 for stable CRC32 functionality on arm64 platforms.
ARG CVCUDA_VERSION=0.17.0
ARG CVCUDA_DEV_SHA256_AMD64=9e2fbdf00af4608280acc54bb5eeb11cc0851c0fe153e6a34c98102cc75ba47a
ARG CVCUDA_DEV_SHA256_ARM64=7e6047ff56275e40de735ae69b1fb23bb4ab85d918dd7d2bba196c37dd92a0a4
ARG CVCUDA_LIB_SHA256_AMD64=70a6033ee6b39d2797b1eb742d633b8970e51992a12c0318df386bb719524bac
ARG CVCUDA_LIB_SHA256_ARM64=5afaeeff7da0487eaf8108b0020c0066ffbebf70bccf81653afc91f081273048
ARG RUST_VERSION=1.85.0
ARG RUSTUP_VERSION=1.29.0
ARG RUSTUP_INIT_SHA256_AMD64=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
ARG RUSTUP_INIT_SHA256_ARM64=9732d6c5e2a098d3521fca8145d826ae0aaa067ef2385ead08e6feac88fa5792

############################################################
# Generic base image
# Notes:
# - setting BASE_IMAGE will override the preset base image
############################################################
ARG BASE_IMAGE=${TARGETARCH}_base
FROM nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 AS amd64_base
FROM nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 AS arm64_base
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

ARG APT_CACHE_ID="holoscan-sdk-apt-cache-${TARGETARCH}"
ARG APT_LIB_ID="holoscan-sdk-apt-lib-${TARGETARCH}"
ARG PIP_CACHE_ID="holoscan-sdk-pip-cache-${TARGETARCH}"

# Install basic tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    apt-get update && apt-get install --no-install-recommends -y \
        curl \
        unzip \
        patch

############################################################
# Set up CUDA APT repo if missing
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
# Python base
############################################################
FROM base AS python-base

# Use Python 3.12 to match the Ubuntu 24.04/DLFW development stack.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
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

# Ensure python is the same as python3
RUN update-alternatives --install /usr/bin/python python $(command -v python3) 1

# Allow future pip installs on the system, no need for a
# venv in our container.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Get recent pip from pypa.io
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

############################################################
# Build tools
############################################################
FROM python-base AS build-tools

# Install build tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
    && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
        | gpg --dearmor -o "$KW_KEYRING" \
    && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake="4.*" \
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

# Install a minimal Rust toolchain for dependencies that build Rust FFI libraries
# during SDK builds, such as HoloMQ's default iceoryx2 backend.
ARG TARGETARCH
ARG RUST_VERSION
ARG RUSTUP_VERSION
ARG RUSTUP_INIT_SHA256_AMD64
ARG RUSTUP_INIT_SHA256_ARM64
ENV RUSTUP_HOME=/opt/rust/rustup
ENV CARGO_HOME=/opt/rust/cargo
ENV PATH=/opt/rust/cargo/bin:${PATH}
RUN case "${TARGETARCH}" in \
        amd64) rust_arch="x86_64"; rustup_sha256="${RUSTUP_INIT_SHA256_AMD64}" ;; \
        arm64) rust_arch="aarch64"; rustup_sha256="${RUSTUP_INIT_SHA256_ARM64}" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -fsSLo /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${rust_arch}-unknown-linux-gnu/rustup-init" \
    && echo "${rustup_sha256}  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && rm -f /tmp/rustup-init \
    && chmod -R a+rX /opt/rust

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
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
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

# The cuda-nsight-systems package is large as it also includes nsys-ui.
# We can install the lighter weight nsight-systems-cli instead, but this requires adding the
# NVIDIA DevTools APT repo.
ARG NSYS_VERSION

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
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
#  libcublas: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libcufft: runtime needed by libtorch, cupy, and headers (-dev) for matx.
#  libcufile: needed by libtorch.
#  libcurand: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libcusolver: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libcusparse: runtime needed by libtorch, cupy-cuda, and headers (-dev) for matx.
#  libnpp-dev: needed by bayer_demosaic, format_converter.
#  libnvjitlink: needed by libtorch, cupy-cuda.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
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
# CV-CUDA dev
############################################################
FROM cuda-toolkit-dev AS cvcuda-dev

ARG TARGETARCH
ARG CVCUDA_VERSION
ARG CVCUDA_DEV_SHA256_AMD64
ARG CVCUDA_DEV_SHA256_ARM64
ARG CVCUDA_LIB_SHA256_AMD64
ARG CVCUDA_LIB_SHA256_ARM64
RUN case "${TARGETARCH}" in \
        amd64) \
            cvcuda_arch="x86_64"; \
            dev_sha256="${CVCUDA_DEV_SHA256_AMD64}"; \
            lib_sha256="${CVCUDA_LIB_SHA256_AMD64}" \
            ;; \
        arm64) \
            cvcuda_arch="aarch64"; \
            dev_sha256="${CVCUDA_DEV_SHA256_ARM64}"; \
            lib_sha256="${CVCUDA_LIB_SHA256_ARM64}" \
            ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && cuda_major=$(echo "${CUDA_VERSION}" | cut -d. -f1) \
    && release_url="https://github.com/CVCUDA/CV-CUDA/releases/download/v${CVCUDA_VERSION}" \
    && curl -fsSL --retry 5 --retry-delay 3 \
        "${release_url}/cvcuda-lib-${CVCUDA_VERSION}-cuda${cuda_major}-${cvcuda_arch}-linux.deb" \
        -o /tmp/cvcuda-lib.deb \
    && curl -fsSL --retry 5 --retry-delay 3 \
        "${release_url}/cvcuda-dev-${CVCUDA_VERSION}-cuda${cuda_major}-${cvcuda_arch}-linux.deb" \
        -o /tmp/cvcuda-dev.deb \
    && echo "${lib_sha256}  /tmp/cvcuda-lib.deb" | sha256sum -c - \
    && echo "${dev_sha256}  /tmp/cvcuda-dev.deb" | sha256sum -c - \
    && apt-get install --no-install-recommends -y \
        /tmp/cvcuda-lib.deb \
        /tmp/cvcuda-dev.deb \
    && rm -f /tmp/cvcuda-lib.deb /tmp/cvcuda-dev.deb

############################################################
# Cudnn dev
############################################################
FROM cvcuda-dev AS cudnn-dev

# Install CUDNN (needed by libtorch)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) \
    && apt-get update \
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

ARG TENSORRT_VERSION
# libnvinfer-safe-headers-dev is a transitive dep of libnvinfer-dev at TRT 10.16+; the pin blocks
# apt from selecting the newer TRT 11 candidate as the resolved dep.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) \
    && apt-get update \
    && TRT_VERSION=$(apt-cache madison libnvinfer10 | grep "${TENSORRT_VERSION}" | grep "+cuda${CUDA_MAJOR}" | head -n 1 | awk '{print $3}') \
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
# Inference dependencies
############################################################################################
FROM tensorrt-dev AS infer-dev

ARG NCCL_VERSION
ARG LIBCUSPARSELT_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) \
    && apt-get update \
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
        ${CONDITIONAL_LIBS} \
    && for pkg in "libcusparselt0-cuda-${CUDA_MAJOR}" "libnvshmem3-cuda-${CUDA_MAJOR}"; do \
        dpkg -L "${pkg}" | grep -E ".so[\\.0-9]+" | xargs dirname | uniq | tee /etc/ld.so.conf.d/${pkg}.conf; \
    done \
    && ldconfig

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
# PyTorch downloader
############################################################
FROM python-base AS pytorch-downloader
ARG TORCH_WHL_DIR=/opt/wheels
ARG PYTORCH_VERSION

RUN --mount=type=cache,target=/root/.cache/pip,id=${PIP_CACHE_ID} \
    CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1-2 --output-delimiter="") \
    && PYTORCH_VERSION="${PYTORCH_VERSION}+cu${CUDA_MAJOR_MINOR}" \
    && python3 -m pip download \
        --dest ${TORCH_WHL_DIR} \
        --pre \
        --no-deps \
        --index-url https://download.pytorch.org/whl \
        torch=="${PYTORCH_VERSION}"

############################################################
# PyTorch (install)
############################################################
FROM infer-dev AS pytorch-dev

# Install PyTorch
# --no-index and --find-links to pull from the local download only.
# --no-deps to skip cuda/nvidia wheel dependencies already present on system for c++ use.
# cleanup_unwanted_libs to remove revendored cuda/nvidia libs.
RUN --mount=type=bind,from=pytorch-downloader,source=/opt/wheels,target=/opt/wheels \
    --mount=type=cache,target=/root/.cache/pip,id=${PIP_CACHE_ID} \
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
RUN --mount=type=cache,target=/root/.cache/pip,id=${PIP_CACHE_ID} \
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
# APT repository configs
############################################################
FROM base AS apt-repo-config

# Setup LLVM APT repository (for newer clang-tidy)
# https://apt.llvm.org/
ARG LLVM_VERSION=18
RUN DISTRO_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] https://apt.llvm.org/${DISTRO_CODENAME}/ llvm-toolchain-${DISTRO_CODENAME}-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list

############################################################
# Vulkan validation layers (built from source)
#
# Ubuntu 24.04's vulkan-validationlayers package is frozen at 1.3.275
# (Vulkan header 275), which predates VK_EXT_present_mode_fifo_latest_ready
# and floods Holoviz runs with false-positive validation errors. No newer
# prebuilt binary exists for arm64 (the LunarG SDK ships x86_64 only), so
# build the layer from source at a pinned SDK tag. UPDATE_DEPS=ON makes the
# build fetch its own matching Vulkan-Headers/SPIRV dependencies.
############################################################
FROM build-tools AS vulkan-validation-layers-builder
ARG VULKAN_VALIDATION_LAYERS_TAG=vulkan-sdk-1.4.321.0
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        pkg-config \
        libx11-dev \
        libxcb1-dev \
        libxrandr-dev \
        libwayland-dev
RUN git clone --depth 1 --branch ${VULKAN_VALIDATION_LAYERS_TAG} \
        https://github.com/KhronosGroup/Vulkan-ValidationLayers.git /tmp/vvl \
    && cmake -S /tmp/vvl -B /tmp/vvl/build -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D UPDATE_DEPS=ON \
        -D CMAKE_INSTALL_LIBDIR=lib \
        -D CMAKE_INSTALL_PREFIX=/opt/vulkan-validation-layers \
    && cmake --build /tmp/vvl/build \
    && cmake --install /tmp/vvl/build \
    && rm -rf /tmp/vvl
# The installed manifest references the library by bare filename, which the
# loader resolves through the system linker search path (where an apt-installed
# copy could shadow this build). Point it at the sibling lib directory instead
# so the manifest and library travel as a self-contained pair.
RUN sed -i \
        's|"library_path": "libVkLayer_khronos_validation.so"|"library_path": "../../../lib/libVkLayer_khronos_validation.so"|' \
        /opt/vulkan-validation-layers/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json \
    && grep -q '"library_path": "\.\./\.\./\.\./lib/libVkLayer_khronos_validation.so"' \
        /opt/vulkan-validation-layers/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json

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
#  clang-tidy-18, clang-tools-18 - static analysis (from LLVM apt repo)
#  abigail-tools - ABI diff utilities, including abidiff
#  xvfb - testing on headless systems
#  libx* - X packages
#  libdrm-dev - Holoviz direct-display DRM ownership (VK_EXT_acquire_drm_display)
#  libvulkan-dev, glslang-tools - for Vulkan apps (Holoviz)
#  (the Vulkan validation layer is built from source instead of the stale
#   Ubuntu 24.04 vulkan-validationlayers package; see vulkan-validation-layers-builder)
#  libwayland-dev, libxkbcommon-dev, pkg-config - GLFW compile dependency for Wayland support
#  libdecor-0-plugin-1-cairo - GLFW runtime dependency for Wayland window decorations
#  libegl1 - to run headless Vulkan apps
#  libv4l-dev - V4L2 operator dependency
#  v4l-utils - V4L2 operator utility
#  libjpeg-turbo8-dev - (8.0) v4l2 mjpeg dependency
ARG LLVM_VERSION=18
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind \
        clang-tidy-${LLVM_VERSION} \
        clang-tools-${LLVM_VERSION} \
        abigail-tools \
        xvfb \
        libx11-dev \
        libxcb-glx0 \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        libdrm-dev \
        libvulkan-dev \
        glslang-tools \
        libwayland-dev \
        libxkbcommon-dev \
        pkg-config \
        libdecor-0-plugin-1-cairo \
        libegl1 \
        libv4l-dev \
        v4l-utils \
        libjpeg-turbo8-dev \
        libtinyxml2-dev \
    && ln -sf /usr/bin/clang-tidy-${LLVM_VERSION} /usr/bin/clang-tidy \
    && ln -sf /usr/bin/clang-query-${LLVM_VERSION} /usr/bin/clang-query

# Vulkan validation layer built from source (see vulkan-validation-layers-builder).
# /usr/local/share/vulkan is ahead of /usr/share/vulkan in the loader's layer
# search path, and the manifest's relative library_path resolves to the copied
# library, so nothing can shadow this build.
COPY --from=vulkan-validation-layers-builder \
    /opt/vulkan-validation-layers/lib/libVkLayer_khronos_validation.so /usr/local/lib/
COPY --from=vulkan-validation-layers-builder \
    /opt/vulkan-validation-layers/share/vulkan/explicit_layer.d/ /usr/local/share/vulkan/explicit_layer.d/

# Workaround: Remove cuda-compat-13-* which is known to not function properly.
# cuda-compat-13-0 does not provide display forward compatibility and does not
# meaningfully support CTK minor version forward compatibility, so we can safely remove it.
# Addresses build container runtime test failures on IGX Thor platforms with IGX-SW 2.0.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) \
    && apt-get purge -y cuda-compat-${CUDA_MAJOR}-0 || true

# Installing libvulkan-dev may re-install the python3 package.
# Re-assert Python 3.12 as the system python.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# PIP INSTALLS
#  requirements.dev.txt
#    coverage - test coverage of python tests
#    pytest*  - testing
#  requirements
#    cloudpickle - dependency for distributed apps
#    cupy-cuda - dependency for holoscan python + examples
#    numpy - dependency for holoscan python + examples
#    pillow - image-processing helper used by SDK examples and validation
#    pip - 22.0.2+ needed for PEP 600 and python wheel symlink resolution
COPY python/requirements.dev.txt /tmp/requirements.dev.txt
COPY python/requirements.txt /tmp/requirements.txt
# Note: --ignore-installed is required to handle a pip/apt conflict on Ubuntu 24.04.
# On Ubuntu 24.04, python3-yaml provides PyYAML 6.0.1 which satisfies >=6.0, but
# when --upgrade triggers an upgrade attempt, pip fails to uninstall the apt-installed
# package because it lacks a RECORD file (apt packages don't create these).
# The error is: "Cannot uninstall PyYAML 6.0.1 ... no RECORD file was found".
# Using --ignore-installed tells pip to install fresh copies without attempting to
# uninstall existing system packages.
RUN --mount=type=cache,target=/root/.cache/pip,id=${PIP_CACHE_ID} \
    python3 -m pip install --upgrade --ignore-installed \
        -r /tmp/requirements.dev.txt \
        -r /tmp/requirements.txt

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


############################################################
# Final stage
############################################################
FROM build AS final
