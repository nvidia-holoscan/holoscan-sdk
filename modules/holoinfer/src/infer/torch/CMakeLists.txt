#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

# Torch ignores CUDA_ARCHITECTURES and sets it to OFF after calling `find_package(Torch)`.
# Instead, it uses TORCH_CUDA_ARCH_LIST to generate the NVCC gencode flags which it appends to
# CUDA_NVCC_FLAGS. The format of that list differs from CUDA_ARCHITECTURES.
# Here we convert CUDA_ARCHITECTURES to the TORCH_CUDA_ARCH_LIST equivalent.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(TORCH_CUDA_ARCH_LIST "Auto")
else()
    set(TORCH_CUDA_ARCH_LIST "")

    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        set(ptx true)

        # Check for suffixes, remove them, and set the PTX flag
        if(arch MATCHES "-virtual$")
            string(REGEX REPLACE "-virtual$" "" arch ${arch})
        elseif(arch MATCHES "-real$")
            string(REGEX REPLACE "-real$" "" arch ${arch})
            set(ptx false)
        endif()

        # Insert decimal point before the last digit.
        # ex: 86 -> 8.6, 120 -> 12.0, 90a -> 9.0a
        string(REGEX REPLACE "([0-9]+)([0-9][a-zA-Z]*)$" "\\1.\\2" arch "${arch}")

        # Append +PTX suffix if needed
        if(ptx)
            set(arch "${arch}+PTX")
        endif()

        list(APPEND TORCH_CUDA_ARCH_LIST ${arch})
    endforeach()
endif()
message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

# Find Torch
find_package(Python3 COMPONENTS Interpreter QUIET)
set(PYTHON_MAJOR_MINOR "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
set(TORCH_SEARCH_DIRS
  "/usr/local/lib/python${PYTHON_MAJOR_MINOR}/dist-packages"
  "/usr/lib/python${PYTHON_MAJOR_MINOR}/dist-packages"
  "/usr/local/lib/python${PYTHON_MAJOR_MINOR}/site-packages"
  "/usr/lib/python${PYTHON_MAJOR_MINOR}/site-packages"
  "$ENV{HOME}/.local/lib/python${PYTHON_MAJOR_MINOR}/site-packages"
  "/opt/nvidia/torch"
)
find_package(Torch 2.8 REQUIRED CONFIG PATHS ${TORCH_SEARCH_DIRS})

# Create the holoinfer torch backend library
add_library(holoinfer_torch SHARED core.cpp)
add_library(holoscan::infer::torch ALIAS holoinfer_torch)
set_target_properties(holoinfer_torch PROPERTIES
    OUTPUT_NAME holoscan_infer_torch
    EXPORT_NAME infer::torch
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    INSTALL_RPATH_USE_LINK_PATH true # find libtorch in install tree
)
target_link_libraries(holoinfer_torch
    PRIVATE
        torch
        holoscan::holoinfer_utils
        holoscan::logger
        yaml-cpp
        holoscan_security_flags
)
target_include_directories(holoinfer_torch
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    )
