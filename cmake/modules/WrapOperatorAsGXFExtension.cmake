# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Generates a Graph Composer registry manifest file to accompany a GXF extension.
#
# The GXF / Graph Composer registry allows GXF-based applications such as Graph Composer to access
# a local or remote cache of GXF extensions. Each extension in the local registry must provide
# a metadata YAML file for inspection.
#
# See also:
# - scripts/generate_gxf_manifest.py
# - https://docs.nvidia.com/metropolis/deepstream/dev-guide/graphtools-docs/docs/text/GraphComposer_Registry.html
#
# Inputs are:
#  MANIFEST_NAME: name of the manifest file to generate
#  EXTENSION_NAME: name of the extension library
#  EXTENSION_TARGET: name of the CMake target for the extension library
#  BINARY_FILES: list of binary files to include in the manifest. Defaults to the library target output.
#  FORWARD_ARGS : Arguments to pass directly to the generation script
#
# Outputs:
#  ${MANIFEST_NAME}: the generated manifest YAML file in the current CMake binary directory.
#                    Default: "{extension_name}_manifest.yaml"
#
# Limitations: The `generate_gxf_manifest.py` utility does not support cross-compilation.
#              If cross-compilation is detected, this function will skip and do nothing.
#
function(generate_gxf_registry_manifest)
  # Skip if HOLOSCAN_ENABLE_GOOGLE_SANITIZER is enabled
  if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
    return()
  endif()
  if(CMAKE_CROSSCOMPILING)
    message(STATUS "Skipping GXF registry manifest generation due to cross-compilation.")
    return()
  endif()

  set(SINGLE_VALUE_VARS
    MANIFEST_NAME
    EXTENSION_NAME
    EXTENSION_TARGET
  )
  set(MULTI_VALUE_VARS
    BINARY_FILES
    FORWARD_ARGS
  )
  cmake_parse_arguments(ARG "" "${SINGLE_VALUE_VARS}" "${MULTI_VALUE_VARS}" ${ARGN})

  if(NOT ARG_MANIFEST_NAME)
    set(ARG_MANIFEST_NAME "${ARG_EXTENSION_NAME}_manifest.yaml")
  endif()
  if(NOT ARG_BINARY_FILES)
    set(ARG_BINARY_FILES "$<TARGET_FILE:${ARG_EXTENSION_TARGET}>")
  endif()

  if(GXF_EXTENSIONS_DIR)
    set(CORE_EXT_SEARCH_PATH "${GXF_EXTENSIONS_DIR}")
  elseif(GXF_DIR)
    get_filename_component(CORE_EXT_SEARCH_PATH "${GXF_DIR}/../../../" ABSOLUTE)
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  find_program(GENERATE_MANIFEST_FILE_PY
    generate_gxf_manifest.py
    HINTS
      "${CMAKE_SOURCE_DIR}/scripts"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../bin"
    REQUIRED
  )
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_MANIFEST_NAME}"
    DEPENDS
      ${ARG_EXTENSION_TARGET}
      ${ARG_BINARY_FILES}
      ${GENERATE_MANIFEST_FILE_PY}
    COMMAND ${Python3_EXECUTABLE} "${GENERATE_MANIFEST_FILE_PY}"
      --output "${CMAKE_CURRENT_BINARY_DIR}/${ARG_MANIFEST_NAME}"
      --name ${ARG_EXTENSION_NAME}
      --extension-library $<TARGET_FILE:${ARG_EXTENSION_TARGET}>
      --arch ${CMAKE_HOST_SYSTEM_PROCESSOR}
      --binaries ${ARG_BINARY_FILES}
      --search-path "${CORE_EXT_SEARCH_PATH}"
      --search-path "${CMAKE_BINARY_DIR}/lib/gxf_extensions"
      --db "${CMAKE_BINARY_DIR}/gxf_extension_cache.pickle"
      --quiet
      ${ARG_FORWARD_ARGS}
  )
  add_custom_target(
    generate_${ARG_EXTENSION_NAME}_gxf_registry_manifest ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${ARG_MANIFEST_NAME}"
    COMMENT "Generating GXF registry manifest for ${ARG_EXTENSION_NAME}"
  )
endfunction()

# Invoke the GXF registry CLI to register a GXF extension manifest to the local cache.
#
# Requirements:
#  - The `registry` executable is available in the system PATH.
#  - Extension dependencies are registered to the local registry cache in advance.
#
# Inputs:
#  EXTENSION_NAME: name of the extension to register
#  MANIFEST: path to the manifest file to register
#  DEPENDS: list of CMake targets or generated files on which the registration depends.
#           Defaults to input manifest file.
#
# Outputs:
#  - `register_${EXTENSION_NAME}` CMake target that registers the extension manifest and can be used
#    as a dependency for subsequent extension registration operations.
#  - register_${EXTENSION_NAME}.stamp: a stamp file indicating the registration target ran.
#
function(register_gxf_extension)
  # Skip if HOLOSCAN_ENABLE_GOOGLE_SANITIZER is enabled
  if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
    return()
  endif()

  cmake_parse_arguments(ARG "" "EXTENSION_NAME;MANIFEST" "DEPENDS" ${ARGN})

  find_program(GXF_REGISTRY_EXECUTABLE registry
    HINTS /usr/bin
    REQUIRED
  )
  if(NOT GXF_REGISTRY_EXECUTABLE)
    message(FATAL_ERROR "Could not find GXF registry executable")
  endif()

  # Mark placeholder file as "dirty" so that the registration target always runs after each reconfigure
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/register_${ARG_EXTENSION_NAME}_configure.stamp" "")
  add_custom_command(
    OUTPUT "register_${ARG_EXTENSION_NAME}.stamp"
    COMMAND ${GXF_REGISTRY_EXECUTABLE} extn add -m "${ARG_MANIFEST}"
    COMMAND ${CMAKE_COMMAND} -E touch "register_${ARG_EXTENSION_NAME}.stamp"
    DEPENDS
      "${ARG_MANIFEST}"
      ${ARG_DEPENDS}
      "${CMAKE_CURRENT_BINARY_DIR}/register_${ARG_EXTENSION_NAME}_configure.stamp"
    COMMENT "Registering ${ARG_EXTENSION_NAME} to the local GXF registry cache"
  )
  add_custom_target(
    register_${ARG_EXTENSION_NAME} ALL
    DEPENDS "register_${ARG_EXTENSION_NAME}.stamp"
    COMMENT "Registering GXF registry manifest for ${ARG_EXTENSION_NAME}"
  )
endfunction()

# Convert UUID hash pair to IETF UUID format
#
# https://tools.ietf.org/html/rfc4122
#
# Inputs:
#   HASH1: first 64-bit hash in format '0x1234567890abcdef'
#   HASH2: second 64-bit hash in format '0x1234567890abcdef'
# Output:
#   String in IETF UUID format '12345678-90ab-cdef-1234-567890abcdef'
function(convert_uuid_hashes_to_ietf HASH1 HASH2 OUTPUT)
  string(REGEX REPLACE "^0x" "" HASH1 "${HASH1}")
  string(REGEX REPLACE "^0x" "" HASH2 "${HASH2}")

  string(SUBSTRING "${HASH1}" 0 8 part1)
  string(SUBSTRING "${HASH1}" 8 4 part2)
  string(SUBSTRING "${HASH1}" 12 4 part3)
  string(SUBSTRING "${HASH2}" 0 4 part4)
  string(SUBSTRING "${HASH2}" 4 12 part5)

  set(${OUTPUT} "${part1}-${part2}-${part3}-${part4}-${part5}" PARENT_SCOPE)
endfunction()

# Generates a GXF extension that includes a GXF codelet which wraps the input C++ operator
#
# Inputs are:
#   OPERATOR_CLASS: name of the native operator class to wrap (with namespace)
#   OPERATOR_HEADER_INCLUDE: path to the native operator class header file
#   OPERATOR_TARGET: cmake target of the library holding the native operator implementation
#   CODELET_ID_HASH[1|2]: hash pair to register the GXF codelet
#   CODELET_NAME: name of the GXF codelet class (without namespace)
#   CODELET_NAMESPACE: namespace of the GXF codelet class
#   CODELET DESCRIPTION: description of the GXF codelet
#   CODELET_TARGET_NAME: name of the cmake target to generate for the gxf codelet library
#     Note: optional, defaults to CODELET_NAME lowercase
#   CODELET_TARGET_PROPERTIES: any valid cmake properties that should affect the target above.
#     Note: `INCLUDE_DIRECTORIES` might be needed to point to the root of the OPERATOR_HEADER_INCLUDE path
#   EXTENSION_ID_HASH[1|2]: hash pair to register the GXF extension
#   EXTENSION_NAME: name of the GXF extension class (without namespace)
#   EXTENSION DESCRIPTION: description of the GXF extension
#   EXTENSION_AUTHOR: GXF extension author
#   EXTENSION_VERSION: GXF extension version
#   EXTENSION_LICENSE: GXF extension license
#   EXTENSION_TARGET_NAME: name of the cmake target to generate for the gxf extension library
#     Note: optional, defaults to EXTENSION_NAME lowercase
#   EXTENSION_TARGET_PROPERTIES: any valid cmake properties that should affect the target above.
#   EXTENSION_DEPENDS: list of GXF extensions on which this extension depends. Accepts generator expressions.
#   MANIFEST_ARGS: list of additional arguments to pass to the manifest generation script
#   REGISTER: whether to automatically register the extension with the local GXF registry cache.
#   REGISTER_DEPENDS: list of CMake targets on which GXF extension registration depends.
#
# Outputs are:
#   lib<CODELET_TARGET_NAME>.so
#   lib<EXTENSION_TARGET_NAME>.so
#   <EXTENSION_TARGET_NAME>_manifest.yaml

function(wrap_operator_as_gxf_extension)
  # Define arguments
  list(APPEND OPTION_VARS REGISTER)
  list(APPEND REQUIRED_SINGLE_VALUE_VARS
    EXTENSION_ID_HASH1
    EXTENSION_ID_HASH2
    EXTENSION_NAME
    EXTENSION_DESCRIPTION
    EXTENSION_AUTHOR
    EXTENSION_VERSION
    EXTENSION_LICENSE
    CODELET_ID_HASH1
    CODELET_ID_HASH2
    CODELET_NAME
    CODELET_NAMESPACE
    CODELET_DESCRIPTION
    OPERATOR_TARGET
    OPERATOR_CLASS
    OPERATOR_HEADER_INCLUDE
  )
  list(APPEND OPTIONAL_SINGLE_VALUE_VARS
    CODELET_TARGET_NAME
    EXTENSION_TARGET_NAME
  )
  list(APPEND MULTI_VALUE_VARS
    EXTENSION_TARGET_PROPERTIES
    CODELET_TARGET_PROPERTIES
    EXTENSION_DEPENDS
    MANIFEST_ARGS
    REGISTER_DEPENDS
  )

  # Parse arguments
  list(APPEND SINGLE_VALUE_VARS
    ${REQUIRED_SINGLE_VALUE_VARS}
    ${OPTIONAL_SINGLE_VALUE_VARS}
  )
  cmake_parse_arguments(
      ARG
      "${OPTION_VARS}"
      "${SINGLE_VALUE_VARS}"
      "${MULTI_VALUE_VARS}"
      ${ARGN}
  )

  # Remove the `ARG` prefix
  foreach(VAR IN LISTS OPTION_VARS SINGLE_VALUE_VARS MULTI_VALUE_VARS)
    if(DEFINED ARG_${VAR})
      set(${VAR} "${ARG_${VAR}}")
    endif()
  endforeach()

  # Check for required arguments
  foreach(VAR IN LISTS REQUIRED_SINGLE_VALUE_VARS)
    if(NOT DEFINED ${VAR})
      message(SEND_ERROR "Missing required ${VAR} argument")
    endif()
  endforeach()

  ## Generate additional variables
  # EXTENSION_CPP
  # CODELET_CPP
  # CODELET_HEADER_INCLUDE
  # CODELET_HEADER
  # CODELET_NAME_UPPERCASE
  # CODELET_NAMESPACE_UPPERCASE
  set(EXTENSION_CPP_SUFFIX)
  if(CODELET_NAME STREQUAL EXTENSION_NAME)
    set(EXTENSION_CPP_SUFFIX "_ext")
  endif()
  set(EXTENSION_CPP ${CMAKE_CURRENT_BINARY_DIR}/${EXTENSION_NAME}${EXTENSION_CPP_SUFFIX}.cpp)
  set(CODELET_CPP ${CMAKE_CURRENT_BINARY_DIR}/${CODELET_NAME}.cpp)
  set(CODELET_HEADER_INCLUDE ${CODELET_NAME}.hpp)
  set(CODELET_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${CODELET_HEADER_INCLUDE})
  string(TOUPPER ${CODELET_NAME} CODELET_NAME_UPPERCASE)
  string(TOUPPER ${CODELET_NAMESPACE} CODELET_NAMESPACE_UPPERCASE)

  # Configure the source files
  set(TEMPLATE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/wrap_operator_as_gxf_template")
  configure_file(${TEMPLATE_DIR}/extension.cpp.in ${EXTENSION_CPP})
  configure_file(${TEMPLATE_DIR}/codelet.cpp.in ${CODELET_CPP})
  configure_file(${TEMPLATE_DIR}/codelet.hpp.in ${CODELET_HEADER})

  # Create codelet library
  if(NOT DEFINED CODELET_TARGET_NAME)
    string(TOLOWER ${CODELET_NAME} CODELET_TARGET_NAME)
  endif()
  message(STATUS ${CODELET_TARGET_NAME})
  add_library(${CODELET_TARGET_NAME} SHARED
    ${CODELET_CPP}
    ${CODELET_HEADER}
  )
  target_link_libraries(${CODELET_TARGET_NAME}
    PUBLIC
      gxf_holoscan_wrapper_lib
      ${OPERATOR_TARGET}
  )

  if(DEFINED CODELET_TARGET_PROPERTIES)
    set_target_properties(${CODELET_TARGET_NAME}
      PROPERTIES ${CODELET_TARGET_PROPERTIES}
    )
  endif()

  # Add link directories property to find the gxf wrapping library from the install tree
  get_target_property(CODELET_LINK_DIRECTORIES ${CODELET_TARGET_NAME} LINK_DIRECTORIES)
  if(NOT CODELET_LINK_DIRECTORIES)
    unset(CODELET_LINK_DIRECTORIES)
  endif()
  list(APPEND CODELET_LINK_DIRECTORIES ${GXF_EXTENSIONS_DIR})
  if(CODELET_LINK_DIRECTORIES)
    set_target_properties(${CODELET_TARGET_NAME} PROPERTIES
                          LINK_DIRECTORIES ${CODELET_LINK_DIRECTORIES})
  endif()

  # Create extension library
  if(NOT DEFINED EXTENSION_TARGET_NAME)
    string(TOLOWER ${EXTENSION_NAME} EXTENSION_TARGET_NAME)
  endif()
  message(STATUS ${EXTENSION_TARGET_NAME})
  add_library(${EXTENSION_TARGET_NAME} SHARED
    ${EXTENSION_CPP}
  )
  target_link_libraries(${EXTENSION_TARGET_NAME}
    PUBLIC ${CODELET_TARGET_NAME}
  )
  if(DEFINED EXTENSION_TARGET_PROPERTIES)
    set_target_properties(${EXTENSION_TARGET_NAME}
      PROPERTIES ${EXTENSION_TARGET_PROPERTIES}
    )
  endif()

  convert_uuid_hashes_to_ietf(${EXTENSION_ID_HASH1} ${EXTENSION_ID_HASH2} CODELET_UUID)
  find_package(CUDAToolkit REQUIRED)
  set(MANIFEST_FORWARD_ARGS
    --uuid "${CODELET_UUID}"
    --version "${EXTENSION_VERSION}"
    --cuda "${CUDAToolkit_VERSION}"
    ${ARG_MANIFEST_ARGS}
  )
  if(ARG_EXTENSION_DEPENDS)
    list(APPEND MANIFEST_FORWARD_ARGS
      "--extension-dependencies"
      ${ARG_EXTENSION_DEPENDS}
    )
  endif()

  generate_gxf_registry_manifest(
    EXTENSION_NAME ${EXTENSION_NAME}
    EXTENSION_TARGET ${EXTENSION_TARGET_NAME}
    MANIFEST_NAME ${EXTENSION_NAME}_manifest.yaml
    BINARY_FILES
      $<TARGET_FILE:${EXTENSION_TARGET_NAME}>
      $<TARGET_FILE:${CODELET_TARGET_NAME}>
    FORWARD_ARGS ${MANIFEST_FORWARD_ARGS}
  )
  if(ARG_REGISTER AND NOT CMAKE_CROSSCOMPILING)
    register_gxf_extension(
      EXTENSION_NAME ${EXTENSION_NAME}
      MANIFEST "${EXTENSION_NAME}_manifest.yaml"
      DEPENDS ${ARG_REGISTER_DEPENDS}
    )
  endif()

  # Add link directories property to find the gxf wrapping library from the install tree
  get_target_property(EXTENSION_LINK_DIRECTORIES ${EXTENSION_TARGET_NAME} LINK_DIRECTORIES)
  # Sets the EXTENSION_LINK_DIRECTORIES to an empty string
  if(NOT EXTENSION_LINK_DIRECTORIES)
    unset(EXTENSION_LINK_DIRECTORIES)
  endif()
  list(APPEND EXTENSION_LINK_DIRECTORIES ${GXF_EXTENSIONS_DIR})
  if(EXTENSION_LINK_DIRECTORIES)
    set_target_properties(${EXTENSION_TARGET_NAME} PROPERTIES
                          LINK_DIRECTORIES ${EXTENSION_LINK_DIRECTORIES})
  endif()
endfunction()
