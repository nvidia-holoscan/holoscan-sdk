# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

set(CMAKE_INSTALL_INCLUDEDIR "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}")
set(CMAKE_INSTALL_LIBDIR "${HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR}")

# Pin NVTX independently of the CUDA Toolkit copy. CUDA 13.0 currently ships NVTX C++ wrapper
# 1.0, which lacks structured-payload helpers used by Holoscan profiler annotations.
set(nvtx3_VERSION 3.3.0)
rapids_cpm_find(nvtx3 ${nvtx3_VERSION}
    GLOBAL_TARGETS nvtx3-c nvtx3-cpp
    BUILD_EXPORT_SET ${HOLOSCAN_DEPEND_EXPORT_SET}
    INSTALL_EXPORT_SET ${HOLOSCAN_DEPEND_EXPORT_SET}

    CPM_ARGS
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.3.0-c-cpp
        GIT_SHALLOW TRUE

    OPTIONS "NVTX3_INSTALL ON"
)
