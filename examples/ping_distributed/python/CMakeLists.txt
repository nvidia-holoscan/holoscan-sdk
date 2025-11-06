# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received message 10: Tensor key: out, shape:' \
        --fail-pattern 'AssertionError:' \
        --fail-pattern 'initialized independent of a parent entity' \
        --fail-pattern 'Unable to convert argument type'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_GPU_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed.py --gpu 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received message 10: Tensor key: out, shape:' \
        --fail-pattern 'AssertionError:' \
        --fail-pattern 'initialized independent of a parent entity' \
        --fail-pattern 'Unable to convert argument type'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_FLOW_TRACKING_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed.py --track 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received message 10: Tensor key: out, shape:' \
        --pass-pattern 'Data Flow Tracking Results' \
        --pass-pattern 'Fragment: fragment1' \
        --pass-pattern 'Fragment: fragment2'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_HELP_STRING_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed.py --help 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'Usage: ping_distributed.py' \
        --fail-pattern 'initialized independent of a parent entity' \
        --fail-pattern 'Unable to convert argument type'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_PING_DISTRIBUTED_INVALID_DEVICE_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'GPUDevice value found and cached. dev_id: -5' \
        --pass-pattern 'cudaSetDevice Failed' \
        --fail-pattern 'initialized independent of a parent entity' \
        --fail-pattern 'Unable to convert argument type'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  # Use a negative id to test with a non-existent device
  set_tests_properties(EXAMPLE_PYTHON_PING_DISTRIBUTED_INVALID_DEVICE_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_UCX_DEVICE_ID=-5"
  )
endif()
