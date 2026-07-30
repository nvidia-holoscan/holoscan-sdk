# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping async buffer application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ping_simple_async_buffer.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple_async_buffer.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple_async_buffer.py"
  COMMENT "Copying ping_simple_async_buffer.py"
)
add_custom_target(python_ping_simple_async_buffer ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ping_simple_async_buffer.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple_async_buffer.py"
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

  add_test(NAME EXAMPLE_PYTHON_PING_SIMPLE_ASYNC_BUFFER_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ping_simple_async_buffer.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
        --pass-pattern 'Rx message value: 20' \
        --pass-pattern 'tx,rx' \
        --pass-pattern 'tx_old,rx'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
