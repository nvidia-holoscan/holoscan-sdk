# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(dlpack 1.0
    GLOBAL_TARGETS dlpack

    CPM_ARGS

    GITHUB_REPOSITORY dmlc/dlpack
    GIT_TAG v1.0
    GIT_SHALLOW TRUE
    PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/dlpack.patch
    EXCLUDE_FROM_ALL
)

# Set 'dlpack_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(dlpack_SOURCE_DIR ${dlpack_SOURCE_DIR} PARENT_SCOPE)
