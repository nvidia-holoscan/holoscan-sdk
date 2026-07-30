# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/native_resource.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/native_resource.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/native_resource.py"
  COMMENT "Copying native_resource.py"
)
add_custom_target(python_native_resource ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/native_resource.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/native_resource.py"
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
  add_test(NAME EXAMPLE_RESOURCES_PYTHON_NATIVE_RESOURCE_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} native_resource.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
        --pass-pattern 'native resource setup method called' \
        --pass-pattern 'MinimalOp compute method called' \
        --fail-pattern 'error' \
        --fail-pattern 'Exception occurred'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
