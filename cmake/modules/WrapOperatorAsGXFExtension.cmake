# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Outputs are:
#   lib<CODELET_TARGET_NAME>.so
#   lib<EXTENSION_TARGET_NAME>.so

function(wrap_operator_as_gxf_extension)
  # Define arguments
  list(APPEND OPTION_VARS "")
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
    if (DEFINED ARG_${VAR})
      set(${VAR} "${ARG_${VAR}}")
    endif()
  endforeach()

  # Check for required arguments
  foreach(VAR IN LISTS REQUIRED_SINGLE_VALUE_VARS)
    if (NOT DEFINED ${VAR})
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
  set(EXTENSION_CPP_SUFFIX )
  if (CODELET_NAME STREQUAL EXTENSION_NAME)
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
  if (NOT DEFINED CODELET_TARGET_NAME)
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
  if (DEFINED CODELET_TARGET_PROPERTIES)
    set_target_properties(${CODELET_TARGET_NAME}
      PROPERTIES ${CODELET_TARGET_PROPERTIES}
    )
  endif()

  # Create extension library
  if (NOT DEFINED EXTENSION_TARGET_NAME)
    string(TOLOWER ${EXTENSION_NAME} EXTENSION_TARGET_NAME)
  endif()
  message(STATUS ${EXTENSION_TARGET_NAME})
  add_library(${EXTENSION_TARGET_NAME} SHARED
    ${EXTENSION_CPP}
  )
  target_link_libraries(${EXTENSION_TARGET_NAME}
    PUBLIC ${CODELET_TARGET_NAME}
  )
  if (DEFINED EXTENSION_TARGET_PROPERTIES)
    set_target_properties(${EXTENSION_TARGET_NAME}
      PROPERTIES ${EXTENSION_TARGET_PROPERTIES}
    )
  endif()

endfunction()
