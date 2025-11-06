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
add_custom_target(python_streaming_with_monitor ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_with_monitor.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_with_monitor.py"
  BYPRODUCTS "streaming_with_monitor.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_with_monitor.py"
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

  add_test(NAME EXAMPLE_PYTHON_STREAMING_WITH_MONITOR_TEST
    COMMAND ${Python3_EXECUTABLE} streaming_with_monitor.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_STREAMING_WITH_MONITOR_TEST PROPERTIES
  PASS_REGULAR_EXPRESSION "gen_signal - Sending value 1.*#iteration: 1.*visualize - Received value 1.*gen_signal - Sending value [2-5].*#iteration: [2-5].*visualize - Received value [2-5].*gen_signal - Sending value [6-9].*#iteration: [6-9].*visualize - Received value [6-9].*gen_signal - Sending value 10.*#iteration: 10.*visualize - Received value 10"
  )
endif()
