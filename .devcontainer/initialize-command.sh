#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Actually, VSCode will run this script in the local workspace folder but
# it's better to be explicit.

# Get 'localWorkspaceFolder' environment variable from the script path.
localWorkspaceFolder=$(git rev-parse --show-toplevel 2> /dev/null || dirname $(dirname $(realpath -s $0)))

# Get the holoscan sdk top directory.
TOP=$(readlink -f "${SCRIPT_DIR}/..")

if [ "${localWorkspaceFolder}" != "${TOP}" ]; then
    echo "Project (git) root is not Holoscan SDK source directory. Copying common-debian.sh from the source folder."
    # In this case, project root is not Holoscan source root.
    # If a file is symlinked, the symlinked file can't be copied to the container by Dockerfile.
    # To prevent error, Copy common-debian.sh from the Holoscan source's .devcontainer folder to
    # the project repository's .devcontainer folder.
    cp -f ${TOP}/.devcontainer/library-scripts/common-debian.sh \
      ${localWorkspaceFolder}/.devcontainer/library-scripts/common-debian.sh
fi

# We need to mount Vulkan icd.d JSON file into the container due to this issue
# (https://github.com/NVIDIA/nvidia-container-toolkit/issues/16).
# However, the location of the file is different depending on the driver installation method:
#   - deb packages: /usr/share/vulkan/icd.d/nvidia_icd.json
#   - .run files: /etc/vulkan/icd.d/nvidia_icd.json
# So we need to check which one exists and mount it.

# Here, we copy the existing icd.d JSON file to the temporary directory (/tmp) and mount it to the
# location (/usr/share/vulkan/icd.d/nvidia_icd.json) that the container expects.
# It is because VSCode DevContainer doesn't support conditional mount points.
# (see https://github.com/microsoft/vscode-remote-release/issues/3972)
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  icd_file=/usr/share/vulkan/icd.d/nvidia_icd.json
elif [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  icd_file=/etc/vulkan/icd.d/nvidia_icd.json
else
  >&2 echo "ERROR: Cannot find the Vulkan ICD file from /usr/share/vulkan/icd.d/nvidia_icd.json or /etc/vulkan/icd.d/nvidia_icd.json"
  exit 1
fi

# Copy the file to the temporary directory with the name 'holoscan_nvidia_icd.json'.
cp ${icd_file} /tmp/holoscan_nvidia_icd.json
echo "Mounting ${icd_file} to /usr/share/vulkan/icd.d/nvidia_icd.json through /tmp/holoscan_nvidia_icd.json"

# Dockerfile in this VSCode DevContainer uses a cache image named `holoscan-sdk-dev` to
# speed up the build process. To rebuild the cache image before container creation, it runs:
#
#   docker buildx use default  # use the default builder to access all the cache images
#   ./run build_image`
#
# as an initialization command.
# It also runs `./run install_gxf` to install the GXF library.
docker buildx use default
${TOP}/run build_image
${TOP}/run install_gxf
