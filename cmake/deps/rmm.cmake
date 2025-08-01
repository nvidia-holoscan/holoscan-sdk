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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

set(rmm_PATCH_FILEPATH "${CMAKE_CURRENT_LIST_DIR}/patches/rmm.diff")

# Copy the patch file to rapids-cmake's patch directory.
# Note: rapids-cmake automatically applies patches from its '${rapids-cmake-dir}/cpm/patches' directory.
# (see https://docs.rapids.ai/api/rapids-cmake/stable/cpm/#rapids-cpm-command-line-controls)
# The patch file extension (.diff or .patch) determines how it's processed.
# (see https://github.com/rapidsai/rapids-cmake/blob/v24.04.00/rapids-cmake/cpm/patches/command_template.cmake.in#L38-L62)
# If the patch contains 'diff' text, it may be skipped silently.
# Check for errors in: <build_dir>/rapids-cmake/patches/rmm/err
if(NOT EXISTS "${rmm_PATCH_FILEPATH}")
  message(FATAL_ERROR "RMM patch file not found: ${rmm_PATCH_FILEPATH}")
endif()
file(COPY ${rmm_PATCH_FILEPATH} DESTINATION ${rapids-cmake-dir}/cpm/patches/rmm)

rapids_cpm_find(rmm 24.04.00
    GLOBAL_TARGETS rmm
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    CPM_ARGS

    GITHUB_REPOSITORY rapidsai/rmm
    GIT_TAG v24.04.00
    # Note: If 'rmm' item is available in 'cmake/deps/rapids-cmake-packages.json', 'PATCH_COMMAND' is
    #       ignored so the following statement is meaningless.
    # PATCH_COMMAND git apply "${rmm_PATCH_FILEPATH}"
    GIT_SHALLOW TRUE

    EXCLUDE_FROM_ALL
)

if(rmm_ADDED)
    # Install the headers needed for development with the SDK
    install(DIRECTORY ${rmm_SOURCE_DIR}/include/rmm
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        )
endif()
