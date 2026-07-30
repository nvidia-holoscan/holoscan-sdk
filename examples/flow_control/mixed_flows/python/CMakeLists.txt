# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mixed_flows_execution.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mixed_flows_execution.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mixed_flows_execution.py"
  COMMENT "Copying mixed_flows_execution.py"
)
add_custom_target(python_mixed_flows_execution ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/mixed_flows_execution.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/mixed_flows_execution.py"
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

  add_test(NAME EXAMPLE_PYTHON_MIXED_FLOWS_EXECUTION_TEST
    COMMAND ${Python3_EXECUTABLE} mixed_flows_execution.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MIXED_FLOWS_EXECUTION_TEST PROPERTIES
  PASS_REGULAR_EXPRESSION "Sending value 1 - node1.*(I am here - alarm.*Received value 1 - node2|Received value 1 - node2.*I am here - alarm).*Sending value 2 - node1.*(I am here - alarm.*Received value 2 - node3|Received value 2 - node3.*I am here - alarm)"
  )
endif()
