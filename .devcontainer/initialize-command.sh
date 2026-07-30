#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

# Dockerfile in this VSCode DevContainer uses a cache image named `holoscan-sdk-build` to
# speed up the build process. To rebuild the cache image before container creation, it runs:
#
#   docker buildx use default  # use the default builder to access all the cache images
#   ./run build_image`
#
# as an initialization command.
docker buildx use default
${TOP}/run build_image

# Tag the built image with a fixed alias so devcontainer.json can reference it
# without requiring the HOLOSCAN_BUILD_IMAGE env var to be set mandatorily.
# This alias always points to whatever was just built for the current platform.
BUILT_IMG_NAME=$(${TOP}/run get_build_img_name)
BUILT_IMG_SHA=$(${TOP}/run get_git_sha)
docker tag "${BUILT_IMG_NAME}:${BUILT_IMG_SHA}" "holoscan-sdk-build-vscode-devcontainer"
