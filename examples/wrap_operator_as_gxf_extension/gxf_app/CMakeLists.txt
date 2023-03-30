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

# For create_gxe_application
include(GenerateGXEApp)

create_gxe_application(
  NAME test_operator_as_gxf_ext
  YAML ping.yaml
  EXTENSIONS
    GXF::std
    gxf_holoscan_wrapper
    gxf_wrapped_ping_rx_native_op
    gxf_wrapped_ping_tx_native_op
  COMPONENT holoscan-examples
)

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
endif()

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_OPERATOR_AS_GXF_EXT_TEST
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_operator_as_gxf_ext
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_OPERATOR_AS_GXF_EXT_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Number of pings received: 10")
endif()
