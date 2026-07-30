# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping_multi_port application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/flow_tracker.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py"
  COMMENT "Copying flow_tracker.py"
)
add_custom_target(python_flow_tracker ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/flow_tracker.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py"
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
  add_test(NAME EXAMPLE_PYTHON_FLOW_TRACKER_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
        --pass-pattern 'Data Flow Tracking Results' \
        --pass-pattern 'root1,middle1,leaf1\\s+' \
        --pass-pattern 'root1,middle1,leaf2\\s+' \
        --pass-pattern 'root2,middle2,leaf2\\s+' \
        --pass-pattern 'root1,middle1\\s+' \
        --pass-pattern 'Current latency: [0-9]+\\.?[0-9]* ms'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
