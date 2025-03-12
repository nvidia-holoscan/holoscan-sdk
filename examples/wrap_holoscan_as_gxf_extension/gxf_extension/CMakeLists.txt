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

include(WrapAsGXFComponent)
include(WrapOperatorAsGXFExtension)

####################################################################################################
# Prepare RPATHs for the example
####################################################################################################

file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(example_library_paths
  ping_variant_custom_native_res
  ping_variant_tx_native_op
  ping_variant_rx_native_op
  gxf_extension
)
set(WRAPPED_TARGETS_INSTALL_RPATH
  "\$ORIGIN:\$ORIGIN/../HoloscanWrapperExtension:\$ORIGIN/${install_lib_relative_path}:\$ORIGIN/${install_lib_relative_path}/gxf_extensions"
)
foreach(example_path ${example_library_paths})
  set(WRAPPED_TARGETS_INSTALL_RPATH "${WRAPPED_TARGETS_INSTALL_RPATH}:\$ORIGIN/../${example_path}")
endforeach()

set(COMMON_TARGET_PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_RPATH "${WRAPPED_TARGETS_INSTALL_RPATH}"
)

####################################################################################################
# Define the GXF extension for the Ping Variant Custom Native Resource and Tx/Rx Operators
####################################################################################################

generate_gxf_resource_wrapper(RESOURCE_HDRS RESOURCE_SRCS EXT_CPP_CONTENT
  RESOURCE_CLASS  myres::PingVarCustomNativeRes
  COMPONENT_NAME  PingVarCustomNativeResComponent
  COMPONENT_NAMESPACE myexts
  COMPONENT_TARGET_NAME gxf_wrapped_ping_variant_custom_native_res_lib
  DESCRIPTION "Ping Variant Custom Native Resource component"
  BRIEF "A custom resource"
  DISPLAY_NAME "Ping Variant Resource"
  HASH1  0xc4c16b8d6ef94a01
  HASH2  0x8014ce5b3e9602b1
  INCLUDE_HEADERS ping_variant_custom_native_res.hpp
  PUBLIC_DEPENDS ping_variant_custom_native_res
  COMPONENT_TARGET_PROPERTIES "${COMMON_TARGET_PROPERTIES}"
)

generate_gxf_operator_wrapper(TX_CODELET_HDRS TX_CODELET_SRCS EXT_CPP_CONTENT
  OPERATOR_CLASS "myops::PingVarTxNativeOp"
  CODELET_NAME PingVarTxNativeOpCodelet
  CODELET_NAMESPACE myexts
  HASH1 0x35545ef8ae1541c5
  HASH2 0x8aef3c2078fc50b4
  CODELET_TARGET_NAME gxf_wrapped_ping_variant_tx_native_op_lib
  DESCRIPTION "Ping Tx Native Operator codelet"
  BRIEF "A custom Ping Tx operator"
  DISPLAY_NAME "Ping Tx Native Operator"
  INCLUDE_HEADERS ping_variant_tx_native_op.hpp
  PUBLIC_DEPENDS ping_variant_tx_native_op
  CODELET_TARGET_PROPERTIES "${COMMON_TARGET_PROPERTIES}"
)

generate_gxf_operator_wrapper(RX_CODELET_HDRS RX_CODELET_SRCS EXT_CPP_CONTENT
  OPERATOR_CLASS "myops::PingVarRxNativeOp"
  CODELET_NAME PingVarRxNativeOpCodelet
  CODELET_NAMESPACE myexts
  HASH1 0x7626bbaa59ec43f7
  HASH2 0xb24d8af9e0a8a74b
  CODELET_TARGET_NAME gxf_wrapped_ping_variant_rx_native_op_lib
  DESCRIPTION "Ping Rx Native Operator codelet"
  BRIEF "A custom Ping Rx operator"
  DISPLAY_NAME "Ping Rx Native Operator"
  INCLUDE_HEADERS ping_variant_rx_native_op.hpp
  PUBLIC_DEPENDS ping_variant_rx_native_op
  CODELET_TARGET_PROPERTIES ${COMMON_TARGET_PROPERTIES}
)

if(HOLOSCAN_REGISTER_GXF_EXTENSIONS)
  set(REGISTER_ARGS
    MAKE_MANIFEST
    REGISTER
  )
endif()

generate_gxf_extension(
  EXTENSION_TARGET_NAME gxf_wrapped_ping_variant_ext
  EXTENSION_NAME PingVarCustomNativeResExtension
  EXTENSION_DESCRIPTION
    "Ping Variant Custom Native extension. Includes wrapped Holoscan custom resource and tx/rx operators"
  EXTENSION_AUTHOR "NVIDIA"
  EXTENSION_VERSION "${HOLOSCAN_BUILD_VERSION}"
  EXTENSION_LICENSE "Apache-2.0"
  EXTENSION_ID_HASH1 0x2b8381ed5c2740a1
  EXTENSION_ID_HASH2 0xbe586c019eaa87be
  INCLUDE_HEADERS
    ${RESOURCE_HDRS}
    ${TX_CODELET_HDRS}
    ${RX_CODELET_HDRS}
  PUBLIC_DEPENDS
    gxf_wrapped_ping_variant_custom_native_res_lib
    gxf_wrapped_ping_variant_tx_native_op_lib
    gxf_wrapped_ping_variant_rx_native_op_lib
  EXTENSION_TARGET_PROPERTIES ${COMMON_TARGET_PROPERTIES}
  ${REGISTER_ARGS}
  EXTENSION_DEPENDS
    $<TARGET_FILE:GXF::std>
    $<TARGET_FILE:GXF::cuda>
    $<TARGET_FILE:gxf_holoscan_wrapper>
  REGISTER_DEPENDS
    register_HoloscanWrapperExtension
  EXT_CPP_CONTENT "${EXT_CPP_CONTENT}"
)
target_link_libraries(gxf_wrapped_ping_variant_ext
  PRIVATE holoscan_security_flags
)
unset(EXT_CPP_CONTENT)


####################################################################################################
# Install the GXF extension and its dependencies
####################################################################################################

# Install following the relative folder path
install(
  TARGETS
    gxf_wrapped_ping_variant_custom_native_res_lib
    gxf_wrapped_ping_variant_tx_native_op_lib
    gxf_wrapped_ping_variant_rx_native_op_lib
    gxf_wrapped_ping_variant_ext
  DESTINATION ${relative_path}
  COMPONENT "holoscan-examples"
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${relative_path}"
  COMPONENT holoscan-examples
)
endif()
