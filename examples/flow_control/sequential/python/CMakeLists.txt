# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sequential_execution.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/sequential_execution.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sequential_execution.py"
  COMMENT "Copying sequential_execution.py"
)
add_custom_target(python_sequential_execution ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/sequential_execution.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/sequential_execution.py"
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

  add_test(NAME EXAMPLE_PYTHON_SEQUENTIAL_EXECUTION_TEST
    COMMAND ${Python3_EXECUTABLE} sequential_execution.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_SEQUENTIAL_EXECUTION_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "I am here - node3"
    FAIL_REGULAR_EXPRESSION "error"
  )
endif()
