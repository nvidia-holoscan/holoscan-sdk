# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Workaround for observed issue where rapids_cpm_find may try to use
# incomplete ucxx-config.cmake output in build dir from previous configuration
set(_SAVED_CMAKE_IGNORE_PREFIX_PATH ${CMAKE_IGNORE_PREFIX_PATH})
list(APPEND CMAKE_IGNORE_PREFIX_PATH "${CMAKE_BINARY_DIR}")

# Disable FetchContent_Populate deprecation warnings for older CPM version
# See: https://cmake.org/cmake/help/latest/policy/CMP0169.html
# TODO: Re-enable this warning when we update rapids-cmake
cmake_policy(SET CMP0169 OLD)
set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)

function(superbuild_depend module_name)
    cmake_parse_arguments(PARSE_ARGV 1 _SUPER_BUILD "" "COMPONENT" "")
    # Set CMAKE_INSTALL_DEFAULT_COMPONENT_NAME before including .cmake file
    # if 'COMPONENT' option is defined
    if(_SUPER_BUILD_COMPONENT)
        message(STATUS "Adding dependent module '${module_name}' to install component '${_SUPER_BUILD_COMPONENT}'")
        set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${_SUPER_BUILD_COMPONENT})
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/deps/${module_name}.cmake")
endfunction()

# Establish the CPM and preset package infrastructure for the project
# (This uses CPM_SOURCE_CACHE and ENV{CPM_SOURCE_CACHE} to cache the downloaded source code)
# https://docs.rapids.ai/api/rapids-cmake/stable/packages/rapids_cpm_versions.html#cpm-version-format
#
# Note: When multiple CPM packages are available, the first one takes precedence.
#       Since matx depends on cccl library via rapids-cmake and Holoscan's rapids-cmake version is
#       old, we need to override cccl library to 2.8.0+.
rapids_cpm_init(OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/rapids-cmake-packages.json")

# Temporarily disable clang-tidy for third-party dependencies to avoid warnings from external code
set(_SAVED_CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")
set(CMAKE_CXX_CLANG_TIDY "")

# Define packages to override the default ones.
superbuild_depend(cccl)
# fmt must be populated before rmm and spdlog to ensure fmt headers are installed in the package
superbuild_depend(fmt_rapids)

# Other dependencies
superbuild_depend(cli11_rapids)
superbuild_depend(cudatoolkit_rapids)
superbuild_depend(concurrent_queue)
superbuild_depend(dlpack_rapids)
superbuild_depend(expected_rapids)
superbuild_depend(glfw_rapids)
superbuild_depend(grpc)
superbuild_depend(hwloc)
superbuild_depend(magic_enum)
superbuild_depend(nvtx3)

# populate spdlog, rapids_logger, RMM in transitive dependency order
# to control fetch, patch, and export behavior
superbuild_depend(spdlog_rapids)
superbuild_depend(rapids_logger)
superbuild_depend(rmm)

superbuild_depend(tensorrt)
superbuild_depend(threads)
superbuild_depend(ucx)
superbuild_depend(v4l2)
superbuild_depend(yaml-cpp_rapids)
superbuild_depend(gxf)
superbuild_depend(eigen3_urm)
superbuild_depend(ucxx_rapids)
superbuild_depend(matx)

# Testing dependencies
if(HOLOSCAN_BUILD_TESTS)
    superbuild_depend(gtest_rapids)
endif()

# Python binding dependencies
if(HOLOSCAN_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    superbuild_depend(pybind11)
endif()

# Add after pybind11 to avoid conflicting pybind11 fetch procedures
if(HOLOSCAN_BUILD_HOLOLINK)
    superbuild_depend(hololink)
endif()

# Restore clang-tidy for project code
set(CMAKE_CXX_CLANG_TIDY "${_SAVED_CMAKE_CXX_CLANG_TIDY}")
unset(_SAVED_CMAKE_CXX_CLANG_TIDY)

set(CMAKE_IGNORE_PREFIX_PATH "${_SAVED_CMAKE_IGNORE_PREFIX_PATH}")

unset(CMAKE_POLICY_DEFAULT_CMP0169)
