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

# Copy native operator round_robin application
add_custom_target(python_round_robin ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/round_robin.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "round_robin.py"
  BYPRODUCTS "round_robin.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/round_robin.py"
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
  add_test(NAME EXAMPLE_PYTHON_ROUND_ROBIN_4_TEST
           COMMAND ${Python3_EXECUTABLE} round_robin.py --threads 4
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_ROUND_ROBIN_4_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "delay03: sending new value \\(100\\)")

  add_test(NAME EXAMPLE_PYTHON_ROUND_ROBIN_0_TEST
           COMMAND ${Python3_EXECUTABLE} round_robin.py --threads 0
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(EXAMPLE_PYTHON_ROUND_ROBIN_0_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "delay03: sending new value \\(100\\)")
endif()
