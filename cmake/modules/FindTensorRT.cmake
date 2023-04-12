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

# Create TensorRT imported targets
#
# This module defines TensorRT_FOUND if the tensorRT include dir is found or
# if the required libraries (COMPONENTS property in find_package)
# are found.
#
# A new imported target is created for each component (library)
# under the TensorRT namespace (TensorRT::${component_name})
#
# Note: this leverages the find-module paradigm [1]. The config-file paradigm [2]
# is recommended instead in CMake.
# [1] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages
# [2] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#find-module-packages

# Find headers
find_path(TensorRT_INCLUDE_DIR NAMES NvInferVersion.h REQUIRED
          PATHS /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu)
mark_as_advanced(TensorRT_INCLUDE_DIR)

# Find version
function(read_version name str)
    string(REGEX MATCH "${name} ([0-9]\\d*)" _ ${str})
    set(${name} ${CMAKE_MATCH_1} PARENT_SCOPE)
endfunction()

file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TRT_VERSION_FILE)
read_version(NV_TENSORRT_MAJOR ${_TRT_VERSION_FILE})
read_version(NV_TENSORRT_MINOR ${_TRT_VERSION_FILE})
read_version(NV_TENSORRT_PATCH ${_TRT_VERSION_FILE})
set(TensorRT_VERSION "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}.${NV_TENSORRT_PATCH}")
unset(_TRT_VERSION_FILE)

# Find libs, and create the imported target
macro(find_trt_library libname)
  if(NOT TARGET TensorRT::${libname})
    find_library(TensorRT_${libname}_LIBRARY NAMES ${libname} REQUIRED)
    mark_as_advanced(TensorRT_${libname}_LIBRARY)
    add_library(TensorRT::${libname} SHARED IMPORTED GLOBAL)
    set_target_properties(TensorRT::${libname} PROPERTIES
        IMPORTED_LOCATION "${TensorRT_${libname}_LIBRARY}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
  endif()
endmacro()

find_trt_library(nvinfer)
find_trt_library(nvinfer_plugin)
find_trt_library(nvcaffe_parser)
find_trt_library(nvonnxparser)
find_trt_library(nvparsers)

# Generate TensorRT_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    FOUND_VAR TensorRT_FOUND
    VERSION_VAR TensorRT_VERSION
    REQUIRED_VARS TensorRT_INCLUDE_DIR # no need for libs/targets, since find_library is REQUIRED
)
