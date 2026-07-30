# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

# Here we are using rapids_cpm_find() function instead of rapids_cpm_spdlog() function
# (https://docs.rapids.ai/api/rapids-cmake/stable/packages/rapids_cpm_spdlog.html), to
# override the default options.
set(version 1.17.0)

include(FetchContent)

# We will ignore the default value of BUILD_SHARED_LIBS for spdlog and
# always build as static with hidden symbols to avoid symbol conflicts
# with other downstream libraries such as ROS.
set(_PREV_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)

set(SPDLOG_GITHUB_REPOSITORY "https://github.com/gabime/spdlog.git")
set(SPDLOG_TAG "v${version}")

set(SPDLOG_FMT_EXTERNAL_HO ON)   # header-only
set(SPDLOG_FMT_EXTERNAL OFF)
set(SPDLOG_INSTALL ON)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY ${SPDLOG_GITHUB_REPOSITORY}
    GIT_TAG        ${SPDLOG_TAG}
    GIT_SHALLOW    TRUE
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(spdlog)

set(BUILD_SHARED_LIBS "${_PREV_BUILD_SHARED_LIBS}")

if(spdlog_ADDED)
    # Hide public spdlog symbols in any Holoscan SDK library to mitigate
    # downstream symbol conflicts.
    target_link_options(spdlog INTERFACE "LINKER:--exclude-libs,libspdlog")

    set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
    set(spdlog_BINARY_DIR "${spdlog_BINARY_DIR}" PARENT_SCOPE)
    set(spdlog_ADDED "${spdlog_ADDED}" PARENT_SCOPE)
    set(spdlog_VERSION ${version} PARENT_SCOPE)
endif()
