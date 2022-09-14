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

list(APPEND CMAKE_MESSAGE_CONTEXT "setupcuda-post")

# If CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE is chained, call the previous one
if(DEFINED HOLOSCAN_PREVIOUS_CMAKE_PROJECT_INCLUDE)
    include("${HOLOSCAN_PREVIOUS_CMAKE_PROJECT_INCLUDE}")
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

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
