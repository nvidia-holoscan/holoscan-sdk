# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

include(FetchContent)

set(RMM_GITHUB_REPOSITORY "https://github.com/rapidsai/rmm.git")
set(RMM_TAG "26.02.00")
set(BUILD_TESTS OFF)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY ${RMM_GITHUB_REPOSITORY}
    GIT_TAG        "v${RMM_TAG}"
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cpp

    PATCH_COMMAND patch -Np1 -i ${CMAKE_CURRENT_LIST_DIR}/patches/rmm.patch
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(rmm)
