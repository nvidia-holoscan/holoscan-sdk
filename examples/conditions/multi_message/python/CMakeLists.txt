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

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy the per-receiver application
add_custom_target(python_multi_message_per_receiver ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_message_per_receiver.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "multi_message_per_receiver.py"
  BYPRODUCTS "multi_message_per_receiver.py"
)

# Copy the sum-of-all application
add_custom_target(python_multi_message_sum_of_all ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_message_sum_of_all.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "multi_message_sum_of_all.py"
  BYPRODUCTS "multi_message_sum_of_all.py"
)

# Copy the sum-of-all application
add_custom_target(python_single_message_timeout ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/single_message_timeout.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "single_message_timeout.py"
  BYPRODUCTS "single_message_timeout.py"
)

# Install the apps
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_message_per_receiver.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_message_sum_of_all.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_message_timeout.py"
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

  add_test(NAME EXAMPLE_PYTHON_MULTI_MESSAGE_PER_RECEIVER_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} multi_message_per_receiver.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'message received on in1: Hello from tx1' \
        --pass-pattern 'message received on in2: Hello from tx2' \
        --pass-pattern 'message received on in3: Hello from tx3' \
        --fail-pattern 'Unable to convert argument type'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_MULTI_MESSAGE_SUM_OF_ALL_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} multi_message_sum_of_all.py 2>&1 | \
       ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
         --pass-pattern 'messages received on in1: \\(tx1, tx1, tx1, tx1' \
         --pass-pattern 'messages received on in2: \\(tx2, tx2, tx2' \
         --pass-pattern 'messages received on in3: \\(tx3' \
         --fail-pattern 'messages received on in3: \\(tx3, tx3, tx3, tx3' \
         --fail-pattern 'Unable to convert argument type'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_SINGLE_MESSAGE_TIMEOUT_TEST
    COMMAND ${Python3_EXECUTABLE} single_message_timeout.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(
    EXAMPLE_PYTHON_SINGLE_MESSAGE_TIMEOUT_TEST PROPERTIES
    # check that despite min_sum=5, only 3 messages arrived due to the execution_frequency
    PASS_REGULAR_EXPRESSION "3 messages received on in"
    FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
  )
endif()
