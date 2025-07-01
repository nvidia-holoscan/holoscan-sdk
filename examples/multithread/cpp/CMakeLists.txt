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

# Create examples
add_executable(multithread
  multithread.cpp
)
target_link_libraries(multithread
  PRIVATE
  holoscan::core
  holoscan::data_loggers::basic_console_logger
)

# Copy config file to the build tree
add_custom_target(multithread_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multithread.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "multithread.yaml"
    BYPRODUCTS "multithread.yaml"
)
add_dependencies(multithread multithread_yaml)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(multithread PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the source
install(FILES multithread.cpp
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
install(TARGETS multithread
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/multithread.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_MULTITHREAD_OPERATOR_TEST
           COMMAND multithread
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_MULTITHREAD_OPERATOR_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "sum of received values: 496")

  # Test the flow tracking
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/multithread.yaml CONFIG_STRING)
  string(REPLACE "tracking: false" "tracking: true" CONFIG_STRING "${CONFIG_STRING}")
  string(REPLACE "data_logging: false" "data_logging: true" CONFIG_STRING "${CONFIG_STRING}")

  # Write the config
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/multithread_tracking.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  add_test(NAME EXAMPLE_CPP_MULTITHREAD_OPERATOR_TRACKING_TEST
           COMMAND multithread ${CONFIG_FILE}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_MULTITHREAD_OPERATOR_TRACKING_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Data Flow Tracking Results:"
                       PASS_REGULAR_EXPRESSION "tx->out: 1"
                       # representative expected data log entries
                       PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:tx.out\\]"
                       PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:rx.values:15\\]"
                       PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:delay15.in\\]"
                       PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:delay15.out_val\\]"
                       PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:delay15.out_name\\]"
                       # rx.names:* expected to be excluded by denylist
                       PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:rx.names:15\\]"
                       PASS_REGULAR_EXPRESSION "Category:Message \\(std::any\\)")
endif()
