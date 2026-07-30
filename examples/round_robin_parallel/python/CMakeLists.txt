# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator round_robin application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/round_robin.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/round_robin.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/round_robin.py"
  COMMENT "Copying round_robin.py"
)
add_custom_target(python_round_robin ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/round_robin.py"
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
