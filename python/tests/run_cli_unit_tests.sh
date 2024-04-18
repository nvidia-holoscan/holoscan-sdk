#!/bin/bash
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

# Error out if a command fails
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PUBLIC_DIR=${SCRIPT_DIR}/../..

get_host_gpu() {
    if ! command -v nvidia-smi >/dev/null; then
        echo "Could not find any GPU drivers on host. Defaulting build to target dGPU/CPU stack."
        echo -n "dgpu"
    elif nvidia-smi  2>/dev/null | grep nvgpu -q; then
        echo -n "igpu"
    else
        echo -n "dgpu"
    fi
}


get_platform_str() {
    local platform="${1:-}"
    local platform_str

    case "${platform}" in
        amd64|x86_64|x86|linux/amd64)
            platform_str="x86_64"
            ;;
        arm64|aarch64|arm|linux/arm64)
            platform_str="aarch64-$(get_host_gpu)"
            ;;
    esac
    echo -n "${platform_str}"
}

ensure_dir_exists() {
    dir=$1
    name=$2
    if [ ! -d "$dir" ]; then
        echo "Required $name directory does not exist: $dir."
        exit 1
    fi
}

platform=$(get_platform_str $(uname -p))
export APP_BUILD_PATH=${PUBLIC_DIR}/build-${platform}/python/lib
export APP_INSTALL_PATH=${PUBLIC_DIR}/install-${platform}/lib
export PYTHONPATH=${APP_BUILD_PATH}:${PYTHONPATH}
export HOLOSCAN_TESTS_DATA_PATH=${PUBLIC_DIR}/tests/data
export LD_LIBRARY_PATH=${APP_INSTALL_PATH}:${LD_LIBRARY_PATH}
ensure_dir_exists $APP_BUILD_PATH "application build"
ensure_dir_exists $APP_INSTALL_PATH "application install"
ensure_dir_exists $HOLOSCAN_TESTS_DATA_PATH "test data"
echo "Running CLI unit test in ${PYTHONPATH}..."
pytest cli/unit --cov ${PUBLIC_DIR}/build-${platform}/python/lib/holoscan/cli --cov-report=xml --cov-report term --capture=tee-sys
