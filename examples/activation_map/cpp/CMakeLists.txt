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

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(activation_map
    activation_map.cpp
)

target_link_libraries(activation_map
  PRIVATE
  holoscan::core
  holoscan::ops::inference
)

# Copy config file
add_custom_target(activation_map_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "activation_map.yaml"
  BYPRODUCTS "activation_map.yaml"
)

add_dependencies(activation_map activation_map_yaml)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(activation_map PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install the compiled example
install(TARGETS activation_map
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES activation_map.cpp
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
endif()

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_ACTIVATION_MAP_TEST
    COMMAND activation_map ${CMAKE_CURRENT_BINARY_DIR}/activation_map.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_ACTIVATION_MAP_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Model first, second, third was selected to infer"
    PASS_REGULAR_EXPRESSION "7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7"
    PASS_REGULAR_EXPRESSION "8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8"
    PASS_REGULAR_EXPRESSION "9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9")
endif()
