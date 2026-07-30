# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(tl-expected 1.1.0
    GLOBAL_TARGETS tl::expected

    CPM_ARGS

    GITHUB_REPOSITORY TartanLlama/expected
    GIT_TAG v1.1.0
    GIT_SHALLOW TRUE
    PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected.patch
    EXCLUDE_FROM_ALL

    OPTIONS
    "EXPECTED_BUILD_TESTS OFF"
    "EXPECTED_BUILD_PACKAGE OFF"
)

# Set 'tl-expected_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(tl-expected_SOURCE_DIR ${tl-expected_SOURCE_DIR} PARENT_SCOPE)
