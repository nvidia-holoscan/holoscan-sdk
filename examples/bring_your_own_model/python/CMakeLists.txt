# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy configuration file into the build tree
add_custom_target(python_byom ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/byom.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "byom.py"
  BYPRODUCTS "byom.py"
)

add_dependencies(python_byom racerx_data)

# Copy configuration file into the build tree
add_custom_target(python_byom_yaml ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/byom.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "byom.yaml"
  BYPRODUCTS "byom.yaml"
)
add_dependencies(python_byom_yaml python_byom)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/byom.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/byom.yaml"
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

  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/byom.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 10" CONFIG_STRING ${CONFIG_STRING})
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/python_byom_config.yaml)
  file(WRITE ${CONFIG_FILE} ${CONFIG_STRING})

  add_test(NAME EXAMPLE_PYTHON_BYOM_TEST
    COMMAND ${Python3_EXECUTABLE} byom.py --config python_byom_config.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_BYOM_TEST PROPERTIES
    DEPENDS "byom.py"
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )

endif()
