# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy Python streaming_with_monitor application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/streaming_with_monitor.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_with_monitor.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_with_monitor.py"
  COMMENT "Copy Python streaming_with_monitor application"
)
add_custom_target(python_streaming_with_monitor ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/streaming_with_monitor.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_with_monitor.py"
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

  add_test(NAME EXAMPLE_PYTHON_STREAMING_WITH_MONITOR_TEST
    COMMAND ${Python3_EXECUTABLE} streaming_with_monitor.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_STREAMING_WITH_MONITOR_TEST PROPERTIES
  PASS_REGULAR_EXPRESSION "gen_signal - Sending value 1.*#iteration: 1.*visualize - Received value 1.*gen_signal - Sending value [2-5].*#iteration: [2-5].*visualize - Received value [2-5].*gen_signal - Sending value [6-9].*#iteration: [6-9].*visualize - Received value [6-9].*gen_signal - Sending value 10.*#iteration: 10.*visualize - Received value 10"
  )
endif()
