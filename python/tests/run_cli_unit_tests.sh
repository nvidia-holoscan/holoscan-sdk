#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

get_platform_str() {
    local platform="${1:-}"
    local platform_str

    case "${platform}" in
        amd64|x86_64|x86|linux/amd64)
            platform_str="amd64"
            ;;
        arm64|aarch64|arm|linux/arm64)
            platform_str="arm64"
            ;;
    esac
    GPU=$(lsmod | grep -q nvgpu && echo "igpu" || echo "dgpu")
    echo -n "${platform_str}-${GPU}"
}

platform=$(get_platform_str $(uname -p))
export PYTHONPATH=${PUBLIC_DIR}/build-${platform}/python/lib:${PYTHONPATH}
export HOLOSCAN_TESTS_DATA_PATH=${PUBLIC_DIR}/tests/data
export HOLOSCAN_INPUT_PATH=${PUBLIC_DIR}/data
echo "Running CLI unit test in ${PYTHONPATH}..."
pytest cli/unit --cov ${PUBLIC_DIR}/build-${platform}/python/lib/holoscan/cli --cov-report=xml --cov-report term --capture=tee-sys