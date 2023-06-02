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

include(WrapOperatorAsGXFExtension)
wrap_operator_as_gxf_extension(
  OPERATOR_CLASS "myops::PingTxNativeOp"
  OPERATOR_HEADER_INCLUDE "ping_tx_native_op/ping_tx_native_op.hpp"
  OPERATOR_TARGET ping_tx_native_op
  CODELET_ID_HASH1 "0x83a6aede926f4a44"
  CODELET_ID_HASH2 "0xbcdf73a7008bdad9"
  CODELET_NAME "PingTxNativeOpCodelet"
  CODELET_NAMESPACE "myexts"
  CODELET_DESCRIPTION "Ping Tx Native Operator codelet"
  CODELET_TARGET_NAME "gxf_wrapped_ping_tx_native_op_lib" # optional, defaults to CODELET_NAME lowercase
  CODELET_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/.. # to find OPERATOR_HEADER_INCLUDE
  EXTENSION_ID_HASH1 "0x2f3f69b27c2c4fd8"
  EXTENSION_ID_HASH2 "0xb119237f5110572d"
  EXTENSION_NAME "PingTxNativeOpExtension"
  EXTENSION_DESCRIPTION "Ping Tx Native Operator extension"
  EXTENSION_AUTHOR "NVIDIA"
  EXTENSION_VERSION "0.5.1"
  EXTENSION_LICENSE "Apache-2.0"
  EXTENSION_TARGET_NAME "gxf_wrapped_ping_tx_native_op" # optional, defaults to EXTENSION_NAME lowercase
  EXTENSION_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

wrap_operator_as_gxf_extension(
  OPERATOR_CLASS "myops::PingRxNativeOp"
  OPERATOR_HEADER_INCLUDE "ping_rx_native_op/ping_rx_native_op.hpp"
  OPERATOR_TARGET ping_rx_native_op
  CODELET_ID_HASH1 "0x3c9db42c37084788"
  CODELET_ID_HASH2 "0x95f931de85a3c5dd"
  CODELET_NAME "PingRxNativeOpCodelet"
  CODELET_NAMESPACE "myexts"
  CODELET_DESCRIPTION "Ping Rx Native Operator codelet"
  CODELET_TARGET_NAME "gxf_wrapped_ping_rx_native_op_lib" # optional, defaults to CODELET_NAME lowercase
  CODELET_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/.. # to find OPERATOR_HEADER_INCLUDE
  EXTENSION_ID_HASH1 "0x2e62c3eec4f04784"
  EXTENSION_ID_HASH2 "0xaed183505e49dc73"
  EXTENSION_NAME "PingRxNativeOpExtension"
  EXTENSION_DESCRIPTION "Ping Rx Native Operator extension"
  EXTENSION_AUTHOR "NVIDIA"
  EXTENSION_VERSION "0.5.1"
  EXTENSION_LICENSE "Apache-2.0"
  EXTENSION_TARGET_NAME "gxf_wrapped_ping_rx_native_op" # optional, defaults to EXTENSION_NAME lowercase
  EXTENSION_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


## Installing ping_tx_native_op

# Set RPATH to find extensions and libs
file(RELATIVE_PATH gxf_holoscan_wrapper_lib_relative_path
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}/gxf_extensions
)
set(ping_tx_native_op_relative_path ../ping_tx_native_op)
set_target_properties(gxf_wrapped_ping_tx_native_op_lib
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN/${ping_tx_native_op_relative_path}:\$ORIGIN/${gxf_holoscan_wrapper_lib_relative_path}"
)
set_target_properties(gxf_wrapped_ping_tx_native_op
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN:\$ORIGIN/${gxf_holoscan_wrapper_lib_relative_path}"
)

# Install following the relative folder path
file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS gxf_wrapped_ping_tx_native_op gxf_wrapped_ping_tx_native_op_lib
  DESTINATION ${relative_path}
  COMPONENT "holoscan-examples"
)


## Installing ping_rx_native_op

# Set RPATH to find extensions and libs
file(RELATIVE_PATH gxf_holoscan_wrapper_lib_relative_path
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}/gxf_extensions
)
set(ping_rx_native_op_relative_path ../ping_rx_native_op)
set_target_properties(gxf_wrapped_ping_rx_native_op_lib
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN/${ping_rx_native_op_relative_path}:\$ORIGIN/${gxf_holoscan_wrapper_lib_relative_path}"
)
set_target_properties(gxf_wrapped_ping_rx_native_op
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN:\$ORIGIN/${gxf_holoscan_wrapper_lib_relative_path}"
)

# Install following the relative folder path
file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS gxf_wrapped_ping_rx_native_op gxf_wrapped_ping_rx_native_op_lib
  DESTINATION "${relative_path}"
  COMPONENT holoscan-examples
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${relative_path}"
  COMPONENT holoscan-examples
)
endif()
