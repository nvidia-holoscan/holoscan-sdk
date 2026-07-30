# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(concurrent_queue 1.0.4
    GLOBAL_TARGETS concurrent_queue
    CPM_ARGS

    GITHUB_REPOSITORY cameron314/concurrentqueue
    GIT_TAG v1.0.4
    GIT_SHALLOW TRUE
    PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/concurrentqueue.patch

    EXCLUDE_FROM_ALL
)

# Set 'concurrent_queue_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(concurrent_queue_SOURCE_DIR ${concurrent_queue_SOURCE_DIR} PARENT_SCOPE)
