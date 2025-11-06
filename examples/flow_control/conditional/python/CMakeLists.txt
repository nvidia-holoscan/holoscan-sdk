# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_custom_target(python_conditional_execution ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/conditional_execution.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "conditional_execution.py"
  BYPRODUCTS "conditional_execution.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/conditional_execution.py"
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

  add_test(NAME EXAMPLE_PYTHON_CONDITIONAL_EXECUTION_TEST
    COMMAND ${Python3_EXECUTABLE} conditional_execution.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CONDITIONAL_EXECUTION_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION ".*I am here - node1.*I am here - node2.*I am here - node3.*I am here - node1.*I am here - node4.*I am here - node5.*"
    FAIL_REGULAR_EXPRESSION "error"
  )
endif()
