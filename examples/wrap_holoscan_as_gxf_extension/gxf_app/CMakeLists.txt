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

# For create_gxe_application
include(GenerateGXEApp)

create_gxe_application(
  NAME test_holoscan_as_gxf_ext
  YAML ping.yaml
  EXTENSIONS
    GXF::std
    GXF::cuda  # required if a GXF component from Holoscan Resource is used
    gxf_holoscan_wrapper
    gxf_wrapped_ping_variant_ext  # generated with the `generate_gxf_extension` CMake function
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
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME EXAMPLE_HOLOSCAN_AS_GXF_EXT_TEST
    COMMAND bash -c
      "${CMAKE_CURRENT_BINARY_DIR}/test_holoscan_as_gxf_ext 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'Number of pings received: 10' \
        --pass-pattern 'PingVarCustomNativeRes: custom_int_type=4, float_type=3.5, numeric=1, numeric_array=\\[10, 20, 30\\], optional_numeric=-1, optional_numeric_array=\\[\\], boolean=true, optional_void_ptr=null, string=string text, optional_resource=null' \
        --pass-pattern 'PingVarTxNativeOp: custom_resource=non-null, numeric=1, numeric_array=\\[10, 20, 30\\], optional_numeric=-1, optional_numeric_array=\\[\\], boolean=true, optional_void_ptr=null, string=string text, optional_resource=null' \
        --fail-pattern 'ERROR' \
        --fail-pattern 'Error' \
        --fail-pattern 'error'"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
