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

# In this file, we:
# 1. Set a default value for CMAKE_CUDA_ARCHITECTURES
# 2. Use enable_language(CUDA) to retrieve CMAKE_CUDA_COMPILER_ID and CMAKE_CUDA_COMPILER_VERSION
# 3. Parse and update CMAKE_CUDA_ARCHITECTURES to override `all` and `all-major` versions
#    as well as create PTX for the last arch only

message(STATUS "Configuring CUDA Architectures")

# Default
# Needed before enable_language(CUDA) or project(... LANGUAGE CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting it to `native`")
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Enable CUDA language
# Generates CMAKE_CUDA_COMPILER_ID and CMAKE_CUDA_COMPILER_VERSION needed below
enable_language(CUDA)

# Function to filter archs and create PTX for last arch
function(update_cmake_cuda_architectures supported_archs warn)
    list(APPEND CMAKE_MESSAGE_CONTEXT "update_cmake_cuda_architectures")

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0.0)
            if("90a" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm90a not supported with nvcc < 12.0.0")
            endif()
            list(REMOVE_ITEM supported_archs "90a")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8.0)
            if("90" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm90 not supported with nvcc < 11.8.0")
            endif()
            if("89" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm89 not supported with nvcc < 11.8.0")
            endif()
            list(REMOVE_ITEM supported_archs "90")
            list(REMOVE_ITEM supported_archs "89")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5.0)
            if("87" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm87 not supported with nvcc < 11.5.0")
            endif()
            list(REMOVE_ITEM supported_archs "87")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.2.0)
            if("86" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm86 not supported with nvcc < 11.2.0")
            endif()
            list(REMOVE_ITEM supported_archs "86")
        endif()
    endif()

    # Create SASS for all architectures in the list and
    # create PTX for the latest architecture for forward-compatibility.
    list(POP_BACK supported_archs latest_arch)
    list(TRANSFORM supported_archs APPEND "-real")
    list(APPEND supported_archs ${latest_arch})

    set(CMAKE_CUDA_ARCHITECTURES ${supported_archs} PARENT_SCOPE)
endfunction()

# CMake "all" and "all-major" have too many CUDA archs, all the way back to sm_20
#   https://gitlab.kitware.com/cmake/cmake/-/blob/master/Modules/CUDA/architectures.cmake
# Rapids does not list/support embedded devices (Xavier sm_72, Orin sm_87, Thor sm_90a)
#   https://github.com/rapidsai/rapids-cmake/blob/branch-23.09/rapids-cmake/cuda/set_architectures.cmake#L60)
# We need to have our own logic to select our own architectures and create PTX for the latest arch
# for forward compatibility. Only keep the default cmake behavior for native archs input.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "all")
    set(CMAKE_CUDA_ARCHITECTURES "60;70;72;75;80;86;87;89;90;90a")
    update_cmake_cuda_architectures("${CMAKE_CUDA_ARCHITECTURES}" FALSE)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major")
    set(CMAKE_CUDA_ARCHITECTURES "60;70;80;90")
    update_cmake_cuda_architectures("${CMAKE_CUDA_ARCHITECTURES}" FALSE)
elseif(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    update_cmake_cuda_architectures("${CMAKE_CUDA_ARCHITECTURES}" TRUE)
endif()

message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
