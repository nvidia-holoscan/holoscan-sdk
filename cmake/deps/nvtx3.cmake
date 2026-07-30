# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

set(nvtx_VERSION 3.3)
rapids_cpm_find(nvtx3 ${nvtx_VERSION}
    GLOBAL_TARGETS nvtx3-c nvtx3-cpp
    CPM_ARGS
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.3.0-c-cpp
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL

    OPTIONS
        NVTX3_INSTALL ON
)
