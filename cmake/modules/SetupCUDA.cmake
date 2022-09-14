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

list(APPEND CMAKE_MESSAGE_CONTEXT "buildconfig")

message(STATUS "Configuring CUDA Architecture")

function(update_cmake_cuda_architectures supported_archs)
    list(APPEND CMAKE_MESSAGE_CONTEXT "update_cmake_cuda_architectures")

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1.0)
        list(REMOVE_ITEM supported_archs "86")
    endif()

    # Create SASS for all architectures in the list and
    # create PTX for the latest architecture for forward-compatibility.
    list(POP_BACK supported_archs latest_arch)
    list(TRANSFORM supported_archs APPEND "-real")
    list(APPEND supported_archs ${latest_arch})

    set(CMAKE_CUDA_ARCHITECTURES ${supported_archs} PARENT_SCOPE)
endfunction()

# Default to 'NATIVE' if no CUDA_ARCHITECTURES are specified
# Otherwise, update CMAKE_CUDA_ARCHITECTURES to the specified values (last architecture creates only PTX)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES is not defined. Using 'NATIVE' to "
        "build only for locally available architecture through rapids_cuda_init_architectures(). "
        "Please specify -DCMAKE_CUDA_ARCHITECTURES='ALL' to support all archs.")
elseif(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL" AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "NATIVE")
    message(STATUS "Requested CUDA architectures are: ${CMAKE_CUDA_ARCHITECTURES}")
    update_cmake_cuda_architectures("${CMAKE_CUDA_ARCHITECTURES}")
endif()

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cuda_init_architectures.html
# - need to be called before project()
rapids_cuda_init_architectures(${HOLOSCAN_PACKAGE_NAME})

# Store the existing project include file path if exists, to chain with the new one
# (https://cmake.org/cmake/help/latest/variable/CMAKE_PROJECT_PROJECT-NAME_INCLUDE.html)
if(DEFINED CMAKE_PROJECT_${HOLOSCAN_PACKAGE_NAME}_INCLUDE)
    set(HOLOSCAN_PREVIOUS_CMAKE_PROJECT_INCLUDE "${CMAKE_PROJECT_${HOLOSCAN_PACKAGE_NAME}_INCLUDE}")
endif()

# Set post-project hook
set(CMAKE_PROJECT_${HOLOSCAN_PACKAGE_NAME}_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/SetupCUDA-post.cmake")

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
