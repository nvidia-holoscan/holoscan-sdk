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

# Generates a bash script that runs GXE with the provided shared libraries
#
# GXF::gxe is an imported target that wasn't configured by CMake. To enable
# running it and providing plugins, we can write a command in a bash script.
#
# The following macro will generate a bash script which:
# - is named after the NAME property
# - passes the YAML property to the --app argument
# - generates a manifest yaml file based on the EXTENSIONS and
#   passes it to the --manifest argument
# - adds GXF::core to the LD_LIBRARY_PATH env variable to resolve runtime
#   linking

# Utility to get a library file path
function(_get_lib_file_path location target)
  get_target_property(imported ${target} IMPORTED)
  if (imported)
    get_target_property(lib ${target} IMPORTED_LOCATION)
  else()
    set(lib $<TARGET_FILE:${target}>)
  endif()
  set(${location} ${lib} PARENT_SCOPE)
endfunction()

# Utility to get a library file directory
function(_get_lib_file_dir location target)
  get_target_property(imported ${target} IMPORTED)
  if (imported)
    get_target_property(lib ${target} IMPORTED_LOCATION)
    get_filename_component(dir ${lib} DIRECTORY)
  else()
    set(dir $<TARGET_FILE_DIR:${target}>)
  endif()
  set(${location} ${dir} PARENT_SCOPE)
endfunction()

# Utility to get all dependency library dirs for a target
function(_get_target_dependency_dirs dep_dirs target)
  if (TARGET ${target})
    # Check for target dependencies
    get_target_property(deps ${target} INTERFACE_LINK_LIBRARIES)
    if (deps)
      foreach(dep IN LISTS deps)
      _get_target_dependency_dirs(${dep_dirs} ${dep})
      endforeach()
    endif()
    # ... and itself
    _get_lib_file_dir(dir ${target})
  else()
    # Not a target: take the dir from the path instead
    get_filename_component(dir ${target} DIRECTORY)
  endif()
  if(NOT ${dir} MATCHES "^/usr*")
    # Only append non system libraries
    list(APPEND ${dep_dirs} ${dir})
  endif()
  set(${dep_dirs} ${${dep_dirs}} PARENT_SCOPE)
endfunction()

function(create_gxe_application)
  # Parse arguments
  list(APPEND GXE_APP_OPTION_VARS "")
  list(APPEND GXE_APP_SINGLE_VALUE_VARS "NAME" "YAML")
  list(APPEND GXE_APP_MULTI_VALUE_VARS "EXTENSIONS")
  cmake_parse_arguments(
    GXE_APP
    "${GXE_APP_OPTION_VARS}"
    "${GXE_APP_SINGLE_VALUE_VARS}"
    "${GXE_APP_MULTI_VALUE_VARS}"
    ${ARGN}
  )

  # Ensure we have GXF::gxe
  if (NOT TARGET GXF::gxe)
    message(FATAL_ERROR "Can't create application without GXF::gxe. Please add gxe to the list of COMPONENTS in find_package(GXF)")
  endif()
  if (NOT TARGET GXF::core)
    message(FATAL_ERROR "Can't create application without GXF::core. Please add core to the list of COMPONENTS in find_package(GXF)")
  endif()

  # Ensure we have the app YAML
  get_filename_component(GXE_APP_YAML ${GXE_APP_YAML} REALPATH)
  if (NOT EXISTS ${GXE_APP_YAML})
    message(FATAL_ERROR "Can't create a GXE application without an app yaml.")
  endif()

  # Ensure we have a NAME
  if (NOT DEFINED GXE_APP_NAME)
    message(FATAL_ERROR "create_gxe_application requires a NAME argument")
  endif()

  # Copy the yaml file
  get_filename_component(GXE_APP_YAML_NAME ${GXE_APP_YAML} NAME)
  cmake_path(APPEND GXE_APP_YAML_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR} ${GXE_APP_YAML_NAME})
  file(COPY ${GXE_APP_YAML} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  # Create manifest file
  list(APPEND GXE_APP_MANIFEST_CONTENT "extensions:")
  if (DEFINED GXE_APP_EXTENSIONS)
    foreach(extension IN LISTS GXE_APP_EXTENSIONS)
      _get_lib_file_path(lib ${extension})
      list(APPEND GXE_APP_MANIFEST_CONTENT ${lib})
      unset(lib)
    endforeach()
  endif()
  list(JOIN GXE_APP_MANIFEST_CONTENT "\n- " GXE_APP_MANIFEST_CONTENT)
  set(GXE_APP_MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}_manifest.yaml)
  file(GENERATE
    OUTPUT ${GXE_APP_MANIFEST}
    CONTENT ${GXE_APP_MANIFEST_CONTENT}
  )

  # Get path to libraries to add to LD_LIBRARY_PATH
  get_target_property(GXF_CORE_LIB GXF::core IMPORTED_LOCATION)
  get_filename_component(GXF_CORE_DIR ${GXF_CORE_LIB} DIRECTORY)
  list(APPEND LIBRARY_PATHS ${GXF_CORE_DIR})
  if (DEFINED GXE_APP_EXTENSIONS)
    foreach(extension IN LISTS GXE_APP_EXTENSIONS)
      _get_target_dependency_dirs(dirs ${extension})
      list(APPEND LIBRARY_PATHS ${dirs})
      unset(dirs)
    endforeach()
  endif()

  # Create bash script
  list(REMOVE_DUPLICATES LIBRARY_PATHS)
  list(JOIN LIBRARY_PATHS ":" LIBRARY_PATHS_STR)
  get_target_property(GXE_PATH GXF::gxe IMPORTED_LOCATION)
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}
    CONTENT
"#!/usr/bin/env bash
export LD_LIBRARY_PATH=${LIBRARY_PATHS_STR}:\"$LD_LIBRARY_PATH\"
${GXE_PATH} --app ${GXE_APP_YAML_BUILD_PATH} --manifest ${GXE_APP_MANIFEST} $@
"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
  )
endfunction()