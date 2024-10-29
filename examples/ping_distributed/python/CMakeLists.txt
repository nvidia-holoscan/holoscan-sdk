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

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping application
add_custom_target(python_ping_distributed ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_distributed.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "ping_distributed.py"
  BYPRODUCTS "ping_distributed.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_distributed.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_TEST
COMMAND python3 ping_distributed.py
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(EXAMPLE_PYTHON_PING_DISTRIBUTED_TEST PROPERTIES
PASS_REGULAR_EXPRESSION "rx received message 10: Tensor key: 'out', shape: \\(32, 64\\)"
FAIL_REGULAR_EXPRESSION "AssertionError:"
FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
)

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_GPU_TEST
    COMMAND python3 ping_distributed.py --gpu
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_DISTRIBUTED_GPU_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "rx received message 10: Tensor key: 'out', shape: \\(32, 64\\)"
    FAIL_REGULAR_EXPRESSION "AssertionError:"
    FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
    FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_FLOW_TRACKING_TEST
    COMMAND python3 ping_distributed.py --track
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_DISTRIBUTED_FLOW_TRACKING_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "rx received message 10: Tensor key: 'out', shape: \\(32, 64\\)"
    PASS_REGULAR_EXPRESSION "Data Flow Tracking Results"
    PASS_REGULAR_EXPRESSION "Fragment: fragment2"
    PASS_REGULAR_EXPRESSION "Fragment: fragment1"
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_HELP_STRING_TEST
    COMMAND python3 ping_distributed.py --help
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_DISTRIBUTED_HELP_STRING_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage: ping_distributed.py \\[OPTIONS\\]"
    FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
    FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_INVALID_DEVICE_TEST
    COMMAND python3 ping_distributed.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # Use a negative id to test with a non-existent device
  set_tests_properties(EXAMPLE_PYTHON_PING_DISTRIBUTED_INVALID_DEVICE_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_UCX_DEVICE_ID=-5"
    PASS_REGULAR_EXPRESSION "GPUDevice value found and cached. dev_id: -5"
    PASS_REGULAR_EXPRESSION "cudaSetDevice Failed - 101, device id -5"
    FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
    FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
  )
endif()
