# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "deps")

# Print current settings
message(STATUS "PROJECT_NAME                     : ${PROJECT_NAME}")
message(STATUS "CMAKE_HOST_SYSTEM                : ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE                 : ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER               : ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER_ID            : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION       : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_FLAGS                  : ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CUDA_COMPILER              : ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID           : ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION      : ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_FLAGS                 : ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_HOST_COMPILER         : ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES         : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR         : ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR         : ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_LIST_DIR           : ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS           : ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_INSTALL_PREFIX             : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_FULL_INCLUDEDIR    : ${CMAKE_INSTALL_FULL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_FULL_LIBDIR        : ${CMAKE_INSTALL_FULL_LIBDIR}")
message(STATUS "CMAKE_MODULE_PATH                : ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH                : ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_PREFIX_PATH (env)          : $ENV{CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_FIND_ROOT_PATH             : ${CMAKE_FIND_ROOT_PATH}")
message(STATUS "CMAKE_LIBRARY_ARCHITECTURE       : ${CMAKE_LIBRARY_ARCHITECTURE}")
message(STATUS "FIND_LIBRARY_USE_LIB64_PATHS     : ${FIND_LIBRARY_USE_LIB64_PATHS}")
message(STATUS "CMAKE_SYSROOT                    : ${CMAKE_SYSROOT}")
message(STATUS "CMAKE_STAGING_PREFIX             : ${CMAKE_STAGING_PREFIX}")
message(STATUS "CMAKE_FIND_ROOT_PATH             : ${CMAKE_FIND_ROOT_PATH}")
message(STATUS "CMAKE_FIND_ROOT_PATH_MODE_INCLUDE: ${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE}")
message(STATUS "")
message(STATUS "BUILD_SHARED_LIBS                : ${BUILD_SHARED_LIBS}")
message(STATUS "HOLOSCAN_BUILD_EXAMPLES          : ${HOLOSCAN_BUILD_EXAMPLES}")
message(STATUS "HOLOSCAN_BUILD_TESTS             : ${HOLOSCAN_BUILD_TESTS}")
message(STATUS "HOLOSCAN_USE_CCACHE              : ${HOLOSCAN_USE_CCACHE}")
message(STATUS "HOLOSCAN_USE_CCACHE_SKIPPED      : ${HOLOSCAN_USE_CCACHE_SKIPPED}")
message(STATUS "HOLOSCAN_CACHE_DIR               : ${HOLOSCAN_CACHE_DIR}")
message(STATUS "HOLOSCAN_TOP                     : ${HOLOSCAN_TOP}")
message(STATUS "HOLOSCAN_INSTALL_LIB_DIR         : ${HOLOSCAN_INSTALL_LIB_DIR}")

# ##############################################################################
# # Basic dependencies
# ##############################################################################
superbuild_depend(threads)

# ##############################################################################
# # RAPIDS dependencies
# ##############################################################################

# Establish the CPM and preset package infrastructure for the project
# (This uses CPM_SOURCE_CACHE and ENV{CPM_SOURCE_CACHE} to cache the downloaded source code)

# https://docs.rapids.ai/api/rapids-cmake/stable/packages/rapids_cpm_versions.html#cpm-version-format
rapids_cpm_init()

superbuild_depend(cudatoolkit_rapids)

superbuild_depend(yaml-cpp_rapids)
superbuild_depend(fmt_rapids)
superbuild_depend(expected_rapids)

superbuild_depend(spdlog_rapids)
# superbuild_depend(rmm_rapids)
superbuild_depend(dlpack_rapids)

# GXF dependencies
superbuild_depend(gxf)
superbuild_depend(glfw_rapids)
superbuild_depend(glad_rapids)
superbuild_depend(tensorrt)
superbuild_depend(ajantv2_rapids)

# Testing dependencies
if(HOLOSCAN_BUILD_TESTS)
    superbuild_depend(gtest_rapids)
endif()
