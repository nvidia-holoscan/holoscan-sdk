# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(FetchContent)

set(CMAKE_SUPERBUILD_DEPS_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

if(NOT COMMAND superbuild_depend)
    function(superbuild_depend module_name)
        cmake_parse_arguments(PARSE_ARGV 1 _SUPER_BUILD "" "COMPONENT" "")

        # Set CMAKE_INSTALL_DEFAULT_COMPONENT_NAME before including .cmake file
        # if 'COMPONENT' option is defined
        if(_SUPER_BUILD_COMPONENT)
            message(STATUS "Adding dependent module '${module_name}' to install component '${_SUPER_BUILD_COMPONENT}'")
            set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${_SUPER_BUILD_COMPONENT})
        endif()

        include("${CMAKE_SUPERBUILD_DEPS_ROOT_DIR}/deps/${module_name}.cmake")
    endfunction()
endif()

# Fetch RAPIDS-cmake
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake
)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
