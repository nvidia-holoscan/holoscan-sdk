# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create GXF imported cmake targets
#
# This module defines GXF_FOUND if all GXF libraries are found or
# if the required libraries (COMPONENTS property in find_package)
# are found.
#
# A new imported target is created for each component (library)
# under the GXF namespace (GXF::${component_name})
#
# Note: this leverages the find-module paradigm [1]. The config-file paradigm [2]
# is recommended instead in CMake.
# [1] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages
# [2] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#find-module-packages

# Define environment
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64)
  set(_internal_GXF_recipe "gxf_x86_64")
  set(_public_GXF_recipe "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
  set(_internal_GXF_recipe "jetpack50")
  set(_public_GXF_recipe "arm64")
else()
  message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} is not an architecture supported by GXF")
endif()

# Library names
list(APPEND _GXF_EXTENSIONS
  behavior_tree
  cuda
  multimedia
  network
  npp
  python_codelet
  sample
  serialization
  std
  tensor_rt
)

# Common headers
find_path(GXF_common_INCLUDE_DIR
  NAMES common/
  REQUIRED
)
mark_as_advanced(GXF_common_INCLUDE_DIR)
list(APPEND GXF_INCLUDE_DIR_VARS GXF_common_INCLUDE_DIR)

# Librairies and their headers
list(APPEND _GXF_LIBRARIES ${_GXF_EXTENSIONS} core)
foreach(component IN LISTS _GXF_LIBRARIES)
  # headers
  find_path(GXF_${component}_INCLUDE_DIR
    NAMES "gxf/${component}/"
  )
  mark_as_advanced(GXF_${component}_INCLUDE_DIR)
  list(APPEND GXF_INCLUDE_DIR_VARS GXF_${component}_INCLUDE_DIR)

  # library
  find_library(GXF_${component}_LIBRARY
    NAMES "gxf_${component}"
    PATH_SUFFIXES
      "${_internal_GXF_recipe}/${component}"
      "${_public_GXF_recipe}/${component}"
  )
  mark_as_advanced(GXF_${component}_LIBRARY)
  list(APPEND GXF_LIBRARY_VARS GXF_${component}_LIBRARY)

  # create imported target
  if(GXF_${component}_LIBRARY AND GXF_${component}_INCLUDE_DIR)
    if(NOT TARGET GXF::${component})
      # Assume SHARED, though technically UNKNOWN since we don't enforce .so
      add_library(GXF::${component} SHARED IMPORTED)
    endif()

    list(APPEND GXF_${component}_INCLUDE_DIRS
      ${GXF_${component}_INCLUDE_DIR}
      ${GXF_common_INCLUDE_DIR}
    )
    set_target_properties(GXF::${component} PROPERTIES
      IMPORTED_LOCATION "${GXF_${component}_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GXF_${component}_INCLUDE_DIRS}"
    )

    set(GXF_${component}_FOUND TRUE)
  else()
    set(GXF_${component}_FOUND FALSE)
  endif()
endforeach()
unset(_GXF_EXTENSIONS)
unset(_GXF_LIBRARIES)

# Find version
if(GXF_core_INCLUDE_DIR)
  # Note: "kGxfCoreVersion \"(.*)\"$" does not work with a simple string
  # REGEX (doesn't stop and EOL, neither $ nor \n), so we first extract
  # the line with file(STRINGS), then the version with string(REGEX)
  file(STRINGS "${GXF_core_INCLUDE_DIR}/gxf/core/gxf.h" _GXF_VERSION_LINE
    REGEX "kGxfCoreVersion"
  )
  string(REGEX MATCH "kGxfCoreVersion \"(.*)\"" _ ${_GXF_VERSION_LINE})
  set(GXF_VERSION ${CMAKE_MATCH_1})
  unset(_GXF_VERSION_LINE)
endif()

# GXE
find_program(GXF_gxe_PATH
  NAMES gxe
  PATH_SUFFIXES
    "${_internal_GXF_recipe}/gxe"
    "${_public_GXF_recipe}/gxe"
)
if(GXF_gxe_PATH)
  if(NOT TARGET GXF::gxe)
    add_executable(GXF::gxe IMPORTED)
  endif()
  set_target_properties(GXF::gxe PROPERTIES
    IMPORTED_LOCATION "${GXF_gxe_PATH}"
  )
  set(GXF_gxe_FOUND TRUE)
else()
  set(GXF_gxe_FOUND FALSE)
endif()

# Generate GXF_FOUND
include(FindPackageHandleStandardArgs)
if(GXF_FIND_COMPONENTS)
  # ... based on requested components/libraries
  find_package_handle_standard_args(GXF
    FOUND_VAR GXF_FOUND
    VERSION_VAR GXF_VERSION
    HANDLE_COMPONENTS # Looks for GXF_${component}_FOUND
  )
else()
  # ... need all the libraries
  find_package_handle_standard_args(GXF
    FOUND_VAR GXF_FOUND
    VERSION_VAR GXF_VERSION
    REQUIRED_VARS ${GXF_INCLUDE_DIR_VARS} ${GXF_LIBRARY_VARS} GXF_gxe_PATH
  )
endif()

# Clean
unset(_internal_GXF_recipe)
unset(_public_GXF_recipe)
