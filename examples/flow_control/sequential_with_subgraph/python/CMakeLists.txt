# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy sequential with subgraph Python application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sequential_with_subgraph.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/sequential_with_subgraph.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sequential_with_subgraph.py"
  COMMENT "Copying sequential_with_subgraph.py"
)
add_custom_target(python_sequential_with_subgraph ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/sequential_with_subgraph.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/sequential_with_subgraph.py"
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

  add_test(NAME EXAMPLE_PYTHON_SEQUENTIAL_WITH_SUBGRAPH_TEST
    COMMAND ${Python3_EXECUTABLE} sequential_with_subgraph.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_SEQUENTIAL_WITH_SUBGRAPH_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "I am here - node4"
    FAIL_REGULAR_EXPRESSION "error"
  )
endif()
