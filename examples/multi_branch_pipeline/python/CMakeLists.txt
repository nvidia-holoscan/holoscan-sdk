# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator multi_branch_pipeline application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/multi_branch_pipeline.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.py"
  COMMENT "Copying multi_branch_pipeline.py"
)
add_custom_target(python_multi_branch_pipeline ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/multi_branch_pipeline.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.py"
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
  # rx2 should receive close to 100 message (values 0 - 99)
  # rx1 will receive only some of these, but it is hard to know exactly which ones so just verify the first
  add_test(NAME EXAMPLE_PYTHON_MULTI_BRANCH_PIPELINE_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/multi_branch_pipeline.py --multi_thread 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'receiver rx2 received value: 90' \
        --pass-pattern 'receiver rx1 received value: 0' \
        --fail-pattern 'Push failed on receiver in'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_MULTI_BRANCH_PIPELINE_EVENT_BASED_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/multi_branch_pipeline.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'receiver rx2 received value: 90' \
        --pass-pattern 'receiver rx1 received value: 0' \
        --fail-pattern 'Push failed on receiver in'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
