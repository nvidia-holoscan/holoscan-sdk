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

#####################################################################################
# Generic Helpers
#####################################################################################

# Wrap `cmake_parse_arguments` to also generate output variables without a prefix.
#
# Warning: this macro may overwrite existing variables from the parent scope with the same name.
#
# Inputs:
#   PREFIX: prefix to use with `cmake_parse_arguments`
#   OPTIONAL_VARS: list of optional arguments
#   SINGLE_VALUE_VARS: list of single-value arguments
#   MULTI_VALUE_VARS: list of multi-value arguments
#   ARGN: arguments to parse
# Outputs: sets variables with and without the prefix.
macro(parse_args_and_strip_prefix PREFIX OPTIONAL_VARS SINGLE_VALUE_VARS MULTI_VALUE_VARS)
  cmake_parse_arguments(
    ${PREFIX}
    "${OPTIONAL_VARS}"
    "${SINGLE_VALUE_VARS}"
    "${MULTI_VALUE_VARS}"
    ${ARGN}
  )

  # Remove the prefix from output variables
  foreach(VAR ${OPTIONAL_VARS} ${SINGLE_VALUE_VARS} ${MULTI_VALUE_VARS})
    if(DEFINED ARG_${VAR})
      set(${VAR} "${ARG_${VAR}}")
    endif()
  endforeach()
endmacro()

# Validate that all required single-value arguments are set.
#
# Inputs:
#   REQUIRED_ARGS: name of the variable storing required arguments
#   PREFIX: optional prefix for input variable checks
# Outputs: error message if any required argument is missing
macro(check_required_args REQUIRED_ARGS PREFIX)
  # Check for required arguments
  if(PREFIX)
    set(PREFIX "${PREFIX}_")
  endif()
  foreach(VAR ${${REQUIRED_ARGS}})
    if(NOT DEFINED ${PREFIX}${VAR})
      message(FATAL_ERROR "Missing required ${VAR} argument")
    endif()
  endforeach()
endmacro()

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

# Update a target to search the GXF extension library directory at link time.
# This allows example builds in the installation tree to succeed when a target is not
# imported for a given GXF extension, such as `-lgxf_holoscan_wrapper`.
#
# Inputs:
#   TARGET_NAME: name of the target to update
macro(set_gxf_extension_link_directory TARGET_NAME)
  # Add link directories property to find the gxf wrapping library from the install tree
  get_target_property(EXTENSION_LINK_DIRECTORIES "${TARGET_NAME}" LINK_DIRECTORIES)
  # If LINK_DIRECTORIES was not found (`NOTFOUND`), reset it to an empty string
  # to cleanly append the extensions directory.
  if(NOT EXTENSION_LINK_DIRECTORIES)
    unset(EXTENSION_LINK_DIRECTORIES)
  endif()
  list(APPEND EXTENSION_LINK_DIRECTORIES ${GXF_EXTENSIONS_DIR})
  if(EXTENSION_LINK_DIRECTORIES)
    set_target_properties("${TARGET_NAME}" PROPERTIES
                          LINK_DIRECTORIES ${EXTENSION_LINK_DIRECTORIES})
  endif()
endmacro()

#####################################################################################
# GXF Extension Wrapper Helpers
#####################################################################################

# Generate and append a macro call to register display information in a GXF extension `.cpp` file.
#
# Inputs:
#   GXF_EXT_STRING_VAR: name of the variable to store the generated string
# Outputs:
#   Appends generated GXF C++ macro call to the variable content.
function(append_gxf_extension_display_info GXF_EXT_STRING_VAR)
  cmake_parse_arguments(ARG "" "DISPLAY_NAME;CATEGORY;BRIEF" "" ${ARGN})

  string(APPEND ${GXF_EXT_STRING_VAR} "
GXF_EXT_FACTORY_SET_DISPLAY_INFO(\"${ARG_DISPLAY_NAME}\", \"${ARG_CATEGORY}\", \"${ARG_BRIEF}\");"
  )
  set(${GXF_EXT_STRING_VAR} ${${GXF_EXT_STRING_VAR}} PARENT_SCOPE)
endfunction()

# Generate and append a macro call to register codelet or resource information in a GXF extension `.cpp` file .
#
# Inputs:
#   GXF_EXT_STRING_VAR: name of the variable to store the generated string
# Outputs:
#   Appends generated GXF C++ macro call to the variable content.
function(append_gxf_extension_factory GXF_EXT_STRING_VAR)
  set(REQUIRED_SINGLE_VALUE_VARS
    HASH1
    HASH2
    TYPE_NAME
  )
  set(OPTIONAL_SINGLE_VALUE_VARS
    BASE_NAME
    BRIEF
    DISPLAY_NAME
    DESCRIPTION
  )
  cmake_parse_arguments(ARG "" "${REQUIRED_SINGLE_VALUE_VARS};${OPTIONAL_SINGLE_VALUE_VARS}" "" ${ARGN})

  # Check for required arguments
  foreach(VAR IN LISTS REQUIRED_SINGLE_VALUE_VARS)
    if(NOT DEFINED ARG_${VAR})
      message(SEND_ERROR "Missing required ${VAR} argument")
    endif()
  endforeach()
  foreach(VAR IN LISTS OPTIONAL_SINGLE_VALUE_VARS)
    if(NOT DEFINED ARG_${VAR})
      set(ARG_${VAR} "")
    endif()
  endforeach()

  if(ARG_BASE_NAME)
    string(APPEND ${GXF_EXT_STRING_VAR} "
GXF_EXT_FACTORY_ADD_VERBOSE(${ARG_HASH1}, ${ARG_HASH2},
                            ${ARG_TYPE_NAME},
                            ${ARG_BASE_NAME},
                            \"${ARG_DISPLAY_NAME}\",
                            \"${ARG_BRIEF}\",
                            \"${ARG_DESCRIPTION}\");"
    )
  else()
    string(APPEND ${GXF_EXT_STRING_VAR} "
GXF_EXT_FACTORY_ADD_0_VERBOSE(${ARG_HASH1}, ${ARG_HASH2},
                      ${ARG_TYPE_NAME},
                      \"${ARG_DISPLAY_NAME}\",
                      \"${ARG_BRIEF}\",
                      \"${ARG_DESCRIPTION}\");"
    )
  endif()
  set(${GXF_EXT_STRING_VAR} ${${GXF_EXT_STRING_VAR}} PARENT_SCOPE)
endfunction()

#####################################################################################
# Holoscan GXF Wrapper Methods
#####################################################################################

# Generates a derived `holoscan::gxf::ResourceWrapper` class that wraps a custom resource type.
#
# A wrapped Holoscan SDK resource can be exposed in a GXF extension to use directly with
# other applications in the GXF ecosystem such as GXE or Graph Composer.
#
# Inputs are:
#   Required:
#   OUTPUT_HEADERS: name of variable to store the generated header filepath
#   OUTPUT_SOURCES: name of variable to store the generated source filepath
#   EXT_CPP_CONTENT_VAR: name of variable to append generated GXF C++ macro content
#   RESOURCE_CLASS: full name of the Holoscan SDK resource class type
#   COMPONENT_NAME: name of the GXF component class to generate (without namespace)
#   COMPONENT_NAMESPACE: namespace of the GXF component class
#   HASH1: first 64-bit hash in format '0x1234567890abcdef'
#   HASH2: second 64-bit hash in format '0x1234567890abcdef'
#
#   Optional:
#   BRIEF: brief description of the GXF component resource
#   COMPONENT_TARGET_NAME: name of the generated component target. Defaults to <lowercase_component_name>.
#   DISPLAY_NAME: display name of the GXF component resource
#   DESCRIPTION: description of the GXF component resource
#   INCLUDE_HEADERS: list of header files to include in the generated component source file.
#      Usually the header file of the custom resource.
#   PUBLIC_DEPENDS: list of targets to link with the generated component. Usually the
#      custom resource library target.
#   COMPONENT_TARGET_PROPERTIES: list of target properties to set for the generated component target,
#      such as a custom output folder.
#
# Outputs are:
#   OUTPUT_HEADERS: relative path to the generated header file in the build directory
#   OUTPUT_SOURCES: relative path to the generated source file in the build directory
#   EXT_CPP_CONTENT_VAR: appended GXF C++ macro content for the generated component
#   Build target to generate the shared library "lib<COMPONENT_NAME>.so" (see COMPONENT_TARGET_NAME)
#
function(generate_gxf_resource_wrapper OUTPUT_HEADERS OUTPUT_SOURCES EXT_CPP_CONTENT_VAR)
  # Define and parse arguments
  set(REQUIRED_SINGLE_VALUE_VARS
    RESOURCE_CLASS
    COMPONENT_NAME
    COMPONENT_NAMESPACE
    HASH1
    HASH2
  )
  set(OPTIONAL_SINGLE_VALUE_VARS
    BRIEF
    COMPONENT_TARGET_NAME
    DESCRIPTION
    DISPLAY_NAME
  )
  set(MULTI_VALUE_VARS
    INCLUDE_HEADERS
    PUBLIC_DEPENDS
    COMPONENT_TARGET_PROPERTIES
  )

  parse_args_and_strip_prefix(ARG
   ""
    "${REQUIRED_SINGLE_VALUE_VARS};${OPTIONAL_SINGLE_VALUE_VARS}"
    "${MULTI_VALUE_VARS}"
    ${ARGN}
  )
  check_required_args(REQUIRED_SINGLE_VALUE_VARS "")

  # Prepare and configure component wrapper files
  set(COMPONENT_CPP_BASENAME ${ARG_COMPONENT_NAME}.cpp)
  set(COMPONENT_CPP ${CMAKE_CURRENT_BINARY_DIR}/${COMPONENT_CPP_BASENAME})
  set(COMPONENT_HEADER_BASENAME ${ARG_COMPONENT_NAME}.hpp)
  set(COMPONENT_HPP ${CMAKE_CURRENT_BINARY_DIR}/${COMPONENT_HEADER_BASENAME})
  list(APPEND COMPONENT_HEADER_INCLUDES "#include \"${COMPONENT_HEADER_BASENAME}\"\n")

  set(COMPONENT_HEADER_INCLUDES "#include \"${COMPONENT_HEADER_BASENAME}\"\n")
  foreach(header ${INCLUDE_HEADERS})
    string(APPEND COMPONENT_HEADER_INCLUDES "#include \"${header}\"\n")
  endforeach()

  string(TOUPPER ${COMPONENT_NAMESPACE} COMPONENT_NAMESPACE_UPPERCASE)
  string(TOUPPER ${COMPONENT_NAME} COMPONENT_NAME_UPPERCASE)
  if(NOT COMPONENT_TARGET_NAME)
    string(TOLOWER ${COMPONENT_NAME} COMPONENT_TARGET_NAME)
  endif()

  set(TEMPLATE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/wrap_resource_as_gxf_template")
  configure_file(${TEMPLATE_DIR}/component.cpp.in ${COMPONENT_CPP})
  configure_file(${TEMPLATE_DIR}/component.hpp.in ${COMPONENT_HPP})

  # Create the component library target
  add_library(${COMPONENT_TARGET_NAME} SHARED
    ${COMPONENT_CPP}
  )
  target_link_libraries(${COMPONENT_TARGET_NAME}
    PUBLIC
      gxf_holoscan_wrapper_lib
      ${PUBLIC_DEPENDS}
  )
  target_include_directories(${COMPONENT_TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
      $<INSTALL_INTERFACE:${HOLOSCAN_INSTALL_INCLUDE_DIR}>
  )

  if(ARG_COMPONENT_TARGET_PROPERTIES)
    set_target_properties(${COMPONENT_TARGET_NAME}
      PROPERTIES ${ARG_COMPONENT_TARGET_PROPERTIES}
    )
  endif()
  set_gxf_extension_link_directory("${COMPONENT_TARGET_NAME}")

  # Generate C++ code for inline factory component calls in the extension `.cpp` file
  append_gxf_extension_factory(
    ${EXT_CPP_CONTENT_VAR}
    HASH1 ${HASH1}
    HASH2 ${HASH2}
    TYPE_NAME "${ARG_COMPONENT_NAMESPACE}::${ARG_COMPONENT_NAME}"
    BASE_NAME "holoscan::gxf::ResourceWrapper"
    DESCRIPTION "${DESCRIPTION}"
    DISPLAY_NAME "${DISPLAY_NAME}"
    BRIEF "${BRIEF}"
  )

  set(${OUTPUT_HEADERS} ${COMPONENT_HEADER_BASENAME} PARENT_SCOPE)
  set(${OUTPUT_SOURCES} ${COMPONENT_CPP_BASENAME} PARENT_SCOPE)
  set(${EXT_CPP_CONTENT_VAR} ${${EXT_CPP_CONTENT_VAR}} PARENT_SCOPE)
endfunction()

# Generates a derived `holoscan::gxf::OperatorWrapper` class that wraps a custom operator type.
#
# A wrapped Holoscan SDK operator can be exposed in a GXF extension to use directly with
# other applications in the GXF ecosystem such as GXE or Graph Composer.
#
# Inputs are:
#   Required:
#   OUTPUT_HEADERS: name of variable to store the generated header filepath
#   OUTPUT_SOURCES: name of variable to store the generated source filepath
#   EXT_CPP_CONTENT_VAR: name of variable to append generated GXF C++ macro content
#   OPERATOR_CLASS: full name of the Holoscan SDK operator class type
#   CODELET_NAME: name of the GXF codelet class to generate (without namespace)
#   CODELET_NAMESPACE: namespace of the GXF codelet class
#   HASH1: first 64-bit hash in format '0x1234567890abcdef'
#   HASH2: second 64-bit hash in format '0x1234567890abcdef'
#
#   Optional:
#   BRIEF: brief description of the GXF codelet operator
#   CODELET_TARGET_NAME: name of the generated codelet target. Defaults to <lowercase_codelet_name>.
#   DESCRIPTION: description of the GXF codelet operator
#   DISPLAY_NAME: display name of the GXF codelet operator
#   INCLUDE_HEADERS: list of header files to include in the generated codelet source file.
#      Usually the header file of the custom operator.
#   PUBLIC_DEPENDS: list of targets to link with the generated codelet. Usually the
#      custom operator library target.
#   CODELET_TARGET_PROPERTIES: list of target properties to set for the generated codelet target,
#      such as a custom output folder.
#
# Outputs:
#   OUTPUT_HEADERS: relative path to the generated header file in the build directory
#   OUTPUT_SOURCES: relative path to the generated source file in the build directory
#   EXT_CPP_CONTENT_VAR: appended GXF C++ macro content for the generated codelet
#   Build target to generate the shared library "lib<CODELET_NAME>.so" (see CODELET_TARGET_NAME)
function(generate_gxf_operator_wrapper OUTPUT_HEADERS OUTPUT_SOURCES EXT_CPP_CONTENT_VAR)
  # Define arguments
  set(REQUIRED_SINGLE_VALUE_VARS
    OPERATOR_CLASS
    CODELET_NAME
    CODELET_NAMESPACE
    HASH1
    HASH2
  )
  set(OPTIONAL_SINGLE_VALUE_VARS
    BRIEF
    CODELET_TARGET_NAME
    DESCRIPTION
    DISPLAY_NAME
  )
  set(MULTI_VALUE_VARS
    INCLUDE_HEADERS
    PUBLIC_DEPENDS
    CODELET_TARGET_PROPERTIES
  )

  parse_args_and_strip_prefix(ARG
    ""
    "${REQUIRED_SINGLE_VALUE_VARS};${OPTIONAL_SINGLE_VALUE_VARS}"
    "${MULTI_VALUE_VARS}"
    ${ARGN}
  )
  check_required_args(REQUIRED_SINGLE_VALUE_VARS "")

  set(CODELET_CPP_BASENAME ${CODELET_NAME}.cpp)
  set(CODELET_CPP ${CMAKE_CURRENT_BINARY_DIR}/${CODELET_CPP_BASENAME})
  set(CODELET_HEADER_BASENAME ${CODELET_NAME}.hpp)
  set(CODELET_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${CODELET_HEADER_BASENAME})

  string(TOUPPER ${CODELET_NAME} CODELET_NAME_UPPERCASE)
  string(TOUPPER ${CODELET_NAMESPACE} CODELET_NAMESPACE_UPPERCASE)
  if(NOT DEFINED CODELET_TARGET_NAME)
    string(TOLOWER ${CODELET_NAME} CODELET_TARGET_NAME)
  endif()

  set(CODELET_HEADER_INCLUDES "#include \"${CODELET_HEADER_BASENAME}\"\n")
  foreach(header ${INCLUDE_HEADERS})
    string(APPEND CODELET_HEADER_INCLUDES "#include \"${header}\"\n")
  endforeach()

  # Configure the source files
  set(TEMPLATE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/wrap_operator_as_gxf_template")
  configure_file(${TEMPLATE_DIR}/codelet.cpp.in ${CODELET_CPP})
  configure_file(${TEMPLATE_DIR}/codelet.hpp.in ${CODELET_HEADER})

  # Create codelet library
  add_library(${CODELET_TARGET_NAME} SHARED ${CODELET_CPP})
  target_include_directories(${CODELET_TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
      $<INSTALL_INTERFACE:${HOLOSCAN_INSTALL_INCLUDE_DIR}>
  )
  target_link_libraries(${CODELET_TARGET_NAME}
    PUBLIC
      gxf_holoscan_wrapper_lib
      ${PUBLIC_DEPENDS}
  )

  if(ARG_CODELET_TARGET_PROPERTIES)
    set_target_properties(${CODELET_TARGET_NAME}
      PROPERTIES ${CODELET_TARGET_PROPERTIES}
    )
  endif()
  set_gxf_extension_link_directory("${CODELET_TARGET_NAME}")

  # Generate C++ code for inline factory component calls in the extension `.cpp` file
  append_gxf_extension_factory(EXT_CPP_CONTENT
    HASH1 ${HASH1}
    HASH2 ${HASH2}
    TYPE_NAME "${CODELET_NAMESPACE}::${CODELET_NAME}"
    BASE_NAME "holoscan::gxf::OperatorWrapper"
    DESCRIPTION "${DESCRIPTION}"
    DISPLAY_NAME "${DISPLAY_NAME}"
    BRIEF "${BRIEF}"
  )

  set(${OUTPUT_HEADERS} ${CODELET_HEADER_BASENAME} PARENT_SCOPE)
  set(${OUTPUT_SOURCES} ${CODELET_CPP_BASENAME} PARENT_SCOPE)
  set(${EXT_CPP_CONTENT_VAR} ${${EXT_CPP_CONTENT_VAR}} PARENT_SCOPE)
endfunction()
