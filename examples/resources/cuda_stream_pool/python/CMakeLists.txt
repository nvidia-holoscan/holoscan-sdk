# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator ping application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cuda_stream.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream.py"
  COMMENT "Copying cuda_stream.py"
)
add_custom_target(python_cuda_stream ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/cuda_stream.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream.py"
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
  add_test(NAME EXAMPLE_PYTHON_CUDA_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_stream.py --count=10 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received non-default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --pass-pattern 'Stream: 0x' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_DEFAULT_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_stream.py --count=10 --default_stream 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --pass-pattern 'Stream: none' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_STREAM_NO_ARRAY_INTERFACE_SYNC_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_stream.py --count=10 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received non-default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --pass-pattern 'Stream: 0x' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_CUDA_STREAM_NO_ARRAY_INTERFACE_SYNC_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG;HOLOSCAN_CUDA_ARRAY_INTERFACE_SYNC=0;CUPY_CUDA_ARRAY_INTERFACE_SYNC=0"
  )

endif()
