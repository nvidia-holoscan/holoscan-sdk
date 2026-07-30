# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(matx 0.9.4
    GLOBAL_TARGETS matx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
    GITHUB_REPOSITORY NVIDIA/MatX
    GIT_TAG v0.9.4
    GIT_SHALLOW TRUE
    PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/patches/matx_install.patch ${CMAKE_CURRENT_LIST_DIR}/patches/matx_setvals.patch
    EXCLUDE_FROM_ALL
    OPTIONS CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON
)

if(matx_ADDED)
    # Correct the include directories for MatX target.
    # MatX's own CMakeLists.txt sets install-interface include directories that are incorrect
    # for the Holoscan SDK's 3rdparty directory layout. We clear them and set the correct ones.
    set_target_properties(matx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(matx INTERFACE
        $<BUILD_INTERFACE:${matx_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/matx>
    )
endif()
