# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

set(CLI11_VERSION 2.5.0)
rapids_cpm_find(CLI11 "${CLI11_VERSION}"
    GLOBAL_TARGETS CLI11::CLI11

    CPM_ARGS
        GITHUB_REPOSITORY CLIUtils/CLI11
        GIT_TAG "v${CLI11_VERSION}"
        GIT_SHALLOW TRUE
        PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/cli11.patch
        EXCLUDE_FROM_ALL

    OPTIONS
        "CLI11_INSTALL ON"
)

# Set 'cli11_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(cli11_SOURCE_DIR ${CLI11_SOURCE_DIR} PARENT_SCOPE)
