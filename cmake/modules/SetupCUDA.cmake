# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

message(STATUS "Configuring CUDA Architectures")

# Default, needed before enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting it to `native`")
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Enable CUDA language (needed for CMAKE_CUDA_COMPILER)
enable_language(CUDA)

message(STATUS "Requested CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# If using "native", nothing else to do. Otherwise, use script to process the request.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    message(STATUS "Using native CUDA architecture detection.")
else()
    # Get the CUDA architectures from the script
    set(_script_command
        "${CMAKE_SOURCE_DIR}/scripts/get_cmake_cuda_archs.py"
        "${CMAKE_CUDA_ARCHITECTURES}"
        "--nvcc-path" "${CMAKE_CUDA_COMPILER}"
        "--min-arch" "70" # Holoscan requirement
        "--verbose"       # Enable debug logging from script
    )
    execute_process(COMMAND ${_script_command}
                    OUTPUT_VARIABLE _script_output
                    ERROR_VARIABLE _script_error
                    RESULT_VARIABLE _script_result
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _script_result EQUAL 0)
        message(FATAL_ERROR "get_cuda_archs.py script failed (${_script_result}):\n${_script_error}")
    else()
        message(DEBUG "\n${_script_error}") # debug verbose logs
        set(CMAKE_CUDA_ARCHITECTURES "${_script_output}")
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()
