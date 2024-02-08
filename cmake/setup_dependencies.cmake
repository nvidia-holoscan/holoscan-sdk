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
rapids_cpm_init()

superbuild_depend(ajantv2_rapids)
superbuild_depend(cli11_rapids)
superbuild_depend(cudatoolkit_rapids)
superbuild_depend(dlpack_rapids)
superbuild_depend(expected_rapids)
superbuild_depend(fmt_rapids)
superbuild_depend(glfw_rapids)
superbuild_depend(grpc)
superbuild_depend(gxf)
superbuild_depend(hwloc)
superbuild_depend(spdlog_rapids)
superbuild_depend(tensorrt)
superbuild_depend(threads)
superbuild_depend(ucx)
superbuild_depend(v4l2)
superbuild_depend(yaml-cpp_rapids)

# Testing dependencies
if(HOLOSCAN_BUILD_TESTS)
    superbuild_depend(gtest_rapids)
endif()

# Python binding dependencies
if(HOLOSCAN_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    superbuild_depend(pybind11)
endif()