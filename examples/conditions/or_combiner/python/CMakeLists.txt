# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy the sum-of-all application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/multi_port_or_combiner.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_port_or_combiner.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/multi_port_or_combiner.py"
  COMMENT "Copying multi_port_or_combiner.py"
)
add_custom_target(python_multi_port_or_combiner ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/multi_port_or_combiner.py"
)

# Install the apps
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_port_or_combiner.py"
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
  add_test(NAME EXAMPLE_PYTHON_MULTI_PORT_OR_COMBINER_TEST
    COMMAND ${Python3_EXECUTABLE} multi_port_or_combiner.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MULTI_PORT_OR_COMBINER_TEST PROPERTIES
    # count > 10 due to the OR condition
    PASS_REGULAR_EXPRESSION "count 15, message received"
    FAIL_REGULAR_EXPRESSION "count 21, message received"
  )
endif()
