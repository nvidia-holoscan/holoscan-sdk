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
# 1. Set up the CMAKE_CUDA_HOST_COMPILER to use CMAKE_CXX_COMPILER if not Clang
# 2. Use enable_language(CUDA) to retrieve CMAKE_CUDA_COMPILER_ID and CMAKE_CUDA_COMPILER_VERSION
# (this requires caching and unsetting CMAKE_CUDA_ARCHITECTURES beforehand to avoid errors)
# 3. Parse and update CMAKE_CUDA_ARCHITECTURES to support NATIVE and ALL architectures, and filter
# out unsupported archs on current host.
#
# Some of this process will be simplified when updating to CMAKE 3.23+ with enhanced support for
# CMAKE_CUDA_ARCHITECTURES
#

list(APPEND CMAKE_MESSAGE_CONTEXT "buildconfig")

message(STATUS "Configuring CUDA Compiler")

# If a toolchain file was provided, include it now to read any CMAKE_CUDA variables it may define.
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif()

# Setup CMAKE_CUDA_HOST_COMPILER
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "Setting CUDA host compiler to use the same CXX compiler defined by CMAKE_CXX_COMPILER(${CMAKE_CXX_COMPILER})")
    # Set host compiler if we are not using clang
    # Clang (>=8) doesn't seem to be compatible with CUDA (>=11.4)
    # - https://forums.developer.nvidia.com/t/cuda-issues-with-clang-compiler/177589
    # - https://forums.developer.nvidia.com/t/building-with-clang-cuda-11-3-0-works-but-with-cuda-11-3-1-fails-regression/182176
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()
endif()

# Keep track of original CMAKE_CUDA_ARCHITECTURES and unset to avoid errors when calling
# 'enable_language(CUDA)' with CMAKE_CUDA_ARCHITECTURES set to 'ALL' or 'NATIVE'
set(CMAKE_CUDA_ARCHITECTURES_CACHE ${CMAKE_CUDA_ARCHITECTURES})
unset(CMAKE_CUDA_ARCHITECTURES)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)

# Delayed CUDA language enablement to make CMAKE_CUDA_COMPILER_ID and CMAKE_CUDA_COMPILER_VERSION
# available for CMAKE_CUDA_ARCHITECTURES setup
enable_language(CUDA)

# Restore original CMAKE_CUDA_ARCHITECTURES
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_CACHE} CACHE STRING "CUDA architectures to build for" FORCE)
unset(CMAKE_CUDA_ARCHITECTURES_CACHE)

message(STATUS "Configuring CUDA Architecture")

# Function to filter archs and create PTX for last arch
function(update_cmake_cuda_architectures supported_archs warn)
    list(APPEND CMAKE_MESSAGE_CONTEXT "update_cmake_cuda_architectures")

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0.0)
            if ("90a" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm90a not supported with nvcc < 12.0.0")
            endif()
            list(REMOVE_ITEM supported_archs "90a")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8.0)
            if ("90" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm90 not supported with nvcc < 11.8.0")
            endif()
            if ("89" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm89 not supported with nvcc < 11.8.0")
            endif()
            list(REMOVE_ITEM supported_archs "90")
            list(REMOVE_ITEM supported_archs "89")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5.0)
            if ("87" IN_LIST supported_archs AND ${warn})
                message(WARNING "sm87 not supported with nvcc < 11.5.0")
            endif()
            list(REMOVE_ITEM supported_archs "87")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.2.0)
            if ("86" IN_LIST supported_archs AND ${warn})
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

# Default to 'NATIVE' if no CUDA_ARCHITECTURES are specified.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES is not defined. Using 'NATIVE' to "
        "build only for locally available architecture through rapids_cuda_set_architectures(). "
        "Please specify -DCMAKE_CUDA_ARCHITECTURES='ALL' to support all archs.")
else()
    message(STATUS "Requested CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
    # Let RAPIDS-CUDA handle "NATIVE"
    # https://docs.rapids.ai/api/rapids-cmake/nightly/command/rapids_cuda_set_architectures.html
    rapids_cuda_set_architectures(NATIVE)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
    # Since `rapids_cuda_init_architectures()` cannot handle all the supported architecture,
    # (only 60;70;75;80;86 are considered. See https://github.com/rapidsai/rapids-cmake/blob/branch-22.08/rapids-cmake/cuda/set_architectures.cmake#L60)
    # we need to have our own logic to add all architectures.
    # FYI, since CMake 3.23, it supports CUDA_ARCHITECTURES (https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
    # Note: 72  is Xavier
    #       87  is Orin
    #       90a is Thor
    set(CMAKE_CUDA_ARCHITECTURES "60;70;72;75;80;86;87;89;90;90a")
    update_cmake_cuda_architectures("${CMAKE_CUDA_ARCHITECTURES}" FALSE)
elseif()
    update_cmake_cuda_architectures("${CMAKE_CUDA_ARCHITECTURES}" TRUE)
endif()

message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
