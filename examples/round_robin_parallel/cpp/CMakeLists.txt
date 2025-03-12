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

# Create examples
add_executable(round_robin
  round_robin.cpp
)
target_link_libraries(round_robin
  PRIVATE
  holoscan::core
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
)

# Copy config file to the build tree
add_custom_target(round_robin_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/round_robin.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "round_robin.yaml"
    BYPRODUCTS "round_robin.yaml"
)
add_dependencies(round_robin round_robin_yaml)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(round_robin PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the source
install(FILES round_robin.cpp
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the compiled example
install(TARGETS round_robin
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/round_robin.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_ROUND_ROBIN_4_TEST
           COMMAND round_robin
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_ROUND_ROBIN_4_TEST PROPERTIES
                        PASS_REGULAR_EXPRESSION "delay03: sending new value \\(100\\)")
endif()
