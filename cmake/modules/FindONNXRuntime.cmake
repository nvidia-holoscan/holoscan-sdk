# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create ONNXRuntime imported cmake targets
#
# This module defines ONNXRuntime_FOUND if all ONNXRuntime libraries are found or
# if the required libraries (COMPONENTS property in find_package)
# are found.
#
# A new imported target is created for each component (library)
# under the ONNXRuntime namespace (ONNXRuntime::${component_name})
#
# Note: this leverages the find-module paradigm [1]. The config-file paradigm [2]
# is recommended instead in CMake.
# [1] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages
# [2] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#find-module-packages

# Find headers
find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime/onnxruntime_c_api.h
)
list(APPEND ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_INCLUDE_DIR}")
list(APPEND ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime")
mark_as_advanced(ONNXRuntime_INCLUDE_DIR)
mark_as_advanced(ONNXRuntime_INCLUDE_DIRS)

# Find version
if(ONNXRuntime_INCLUDE_DIR)
  file(STRINGS "${ONNXRuntime_INCLUDE_DIR}/../VERSION_NUMBER" ONNXRuntime_VERSION)
endif()

# Find libraries
find_library(ONNXRuntime_LIBRARY NAMES onnxruntime)
mark_as_advanced(ONNXRuntime_LIBRARY)
add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
   IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
   INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
)

# Generate ONNXRuntime_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    FOUND_VAR ONNXRuntime_FOUND
    VERSION_VAR ONNXRuntime_VERSION
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)
