# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ping_periodic_async_buffer.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_async_buffer.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_async_buffer.py"
  COMMENT "Copying ping_periodic_async_buffer.py"
)
add_custom_target(python_ping_periodic_async_buffer ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ping_periodic_async_buffer.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_async_buffer.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME EXAMPLE_PYTHON_PING_PERIODIC_ASYNC_BUFFER
    COMMAND ${Python3_EXECUTABLE} ping_periodic_async_buffer.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_PERIODIC_ASYNC_BUFFER PROPERTIES
    PASS_REGULAR_EXPRESSION "Rx message received: 10"
  )
endif()
