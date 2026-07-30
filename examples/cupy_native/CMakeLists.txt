# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator matmul application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/matmul.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/matmul.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/matmul.py"
  COMMENT "Copying matmul.py"
)
add_custom_target(python_matmul ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/matmul.py"
)

# Install the app and doc
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/matmul.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
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
  add_test(NAME EXAMPLE_PYTHON_MATMUL_TEST
    COMMAND ${Python3_EXECUTABLE} matmul.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MATMUL_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Scheduler finished.")
endif()
