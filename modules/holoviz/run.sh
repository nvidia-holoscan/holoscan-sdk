#!/bin/bash -e
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

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

COMMAND=""

TARGET_ARCH=${TARGET_ARCH:-$(uname -m)}
CUDA_VERSION="11.4.0"
DISTRIB_ID="ubuntu"
DISTRIB_RELEASE="20.04"

# colors
RED="\033[0;31m"
GREEN="\033[0;32m"
NC="\033[0m" # No Color

# print a failure message
function failMsg
{
    echo -e "$RED$1$NC$2"
}

function image_build_dev
{
    # CUDA base images
    if [ ${TARGET_ARCH} == "aarch64" ]; then
        CUDA_BASE_IMAGE=nvidia/cuda-arm64
    else
        CUDA_BASE_IMAGE=nvidia/cuda
    fi
    CUDA_IMAGE=${CUDA_BASE_IMAGE}:${CUDA_VERSION}-devel-${DISTRIB_ID}${DISTRIB_RELEASE}

    ${DOCKER_CMD} build --network host ${DOCKER_BUILD_ARGS} --build-arg CUDA_IMAGE=${CUDA_IMAGE} --build-arg TARGET_ARCH=${TARGET_ARCH} \
        . -f ./docker/Dockerfile.dev \
        -t ${DOCKER_DEV_IMAGE}:${DEV_IMAGE_TAG} $@
}

function build
{
    ${DOCKER_CMD} run --network host --rm ${DOCKER_INTERACTIVE} ${DOCKER_TTY} -u $(id -u):$(id -g) \
        -v $PWD:$PWD \
        -w $PWD \
        -e CLARA_HOLOVIZ_VERSION=${CLARA_HOLOVIZ_VERSION} \
        ${DOCKER_DEV_IMAGE}:${DEV_IMAGE_TAG} ./scripts/build.sh -o ${BUILD_DIR} $@
}

while (($#)); do
case $1 in
    -h|--help)
    echo "Usage: $0 COMMAND [-h|--help]"
    echo ""
    echo " -h, --help"
    echo "  display this help message"
    echo " --image_build_dev"
    echo "  build the development docker container"
    echo " --build"
    echo "  build Holoviz"
    exit 1
    ;;
    image_build_dev|build)
    COMMAND=$1
    shift
    break
    ;;
    *)
    failMsg "Unknown option '"$1"'"
    exit 1
    ;;
esac
done

SDK_TOP=$PWD

# version
CLARA_HOLOVIZ_VERSION="$(cat ${SDK_TOP}/VERSION)"

# build output directory
BUILD_DIR=${BUILD_DIR:-build_${TARGET_ARCH}}

# docker dev image tag
DEV_IMAGE_TAG="$(cat ${SDK_TOP}/docker/DEV_VERSION)"
# docker dev image name
DOCKER_DEV_IMAGE=clara-holoviz-dev_${TARGET_ARCH}

# check if the current user is part of the docker group
# and we can run docker without sudo
if id | grep &>/dev/null '\bdocker\b' || ! command -v sudo >/dev/null; then
    DOCKER_CMD=docker
else
    echo "The current user is not in the 'docker' group, running 'docker' with 'sudo'"
    DOCKER_CMD="sudo docker"
fi

DOCKER_BUILD_ARGS="--pull"

# enable terminal usage to be able to Ctrl-C a build or test
if [[ ! -z ${JENKINS_URL} ]]; then
    # there is no terminal when running in Jenkins
    DOCKER_INTERACTIVE=
    DOCKER_TTY=-t
    # always build official image without cache
    DOCKER_BUILD_ARGS="${DOCKER_BUILD_ARGS} --no-cache"
else
    DOCKER_INTERACTIVE=-i
    DOCKER_TTY=-t
fi

${COMMAND} $@
